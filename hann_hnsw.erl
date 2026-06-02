%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — HNSW Index  (gen_server, ETS-backed)
%%%
%%% ── Read/Write split ────────────────────────────────────────────────
%%%
%%%   search/2  ─ NOT a gen_server call.
%%%               Reads hann_idx_state + hann_nodes ETS tables directly.
%%%               N Erlang schedulers can execute N parallel searches
%%%               with zero serialisation — no lock, no mailbox hop.
%%%
%%%   add/2     ─ gen_server:call  (serialised).
%%%               Maintains HNSW graph invariants, updates ETS, then
%%%               casts to hann_scylla for async ScyllaDB persistence.
%%%
%%% ── State ───────────────────────────────────────────────────────────
%%%
%%%   The gen_server #state holds only index metadata (dim, M, ef, …).
%%%   The actual node graph lives entirely in the hann_nodes ETS table
%%%   owned by hann_ets, so a crash of this gen_server loses nothing —
%%%   on restart it re-reads entry + max_layer from hann_idx_state ETS
%%%   (already hydrated by hann_scylla during its own init).
%%%-------------------------------------------------------------------
-module(hann_hnsw).
-behaviour(gen_server).

-export([start_link/1, add/2, search/2, bulk_add/1, stats/0]).
-export([init/1, handle_call/3, handle_cast/2,
         handle_info/2, terminate/2, code_change/3]).

-record(state, {
    dim       :: pos_integer(),
    m         :: pos_integer(),
    m0        :: pos_integer(),   %% 2 × M  (layer-0 max-degree)
    ef        :: pos_integer(),
    ml        :: float(),         %% 1 / ln(M)
    dist_fn   :: cosine | euclidean,
    entry     :: undefined | non_neg_integer(),
    max_layer :: integer()
}).

%%%-------------------------------------------------------------------
%%% Public API
%%%-------------------------------------------------------------------

start_link(Opts) ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, Opts, []).

%%% ── CONCURRENT READ PATH ───────────────────────────────────────────
%%% This function never touches the gen_server mailbox.
%%% Multiple callers (HTTP workers, hann_grg) execute it in parallel
%%% across all schedulers — pure ETS reads with read_concurrency:true.

-spec search([float()], pos_integer()) -> [{float(), non_neg_integer()}].
search(QueryVec, K) ->
    try
        case hann_ets:get_state(entry, undefined) of
            undefined ->
                [];
            Entry ->
                MaxLayer = hann_ets:get_state(max_layer, 0),
                DistFn   = hann_ets:get_state(dist_fn,   cosine),
                Ef       = hann_ets:get_state(ef,        64),
                do_search(QueryVec, K, Entry, MaxLayer, Ef, DistFn)
        end
    catch
        %% ETS table not yet initialised (race on startup)
        error:badarg -> []
    end.

%%% ── SERIALISED WRITE PATH ──────────────────────────────────────────

-spec add(non_neg_integer(), [float()]) -> ok | {error, term()}.
add(NodeID, Vector) ->
    gen_server:call(?MODULE, {add, NodeID, Vector}, 30_000).

-spec bulk_add([{non_neg_integer(), [float()]}]) -> {ok, non_neg_integer()}.
bulk_add(Pairs) ->
    gen_server:call(?MODULE, {bulk_add, Pairs}, 300_000).

-spec stats() -> map().
stats() ->
    gen_server:call(?MODULE, stats, 5_000).

%%%-------------------------------------------------------------------
%%% gen_server callbacks
%%%-------------------------------------------------------------------

init(Opts) ->
    M      = maps:get(m,       Opts, 16),
    Ef     = maps:get(ef,      Opts, 64),
    Dim    = maps:get(dim,     Opts, 128),
    DistFn = maps:get(dist_fn, Opts, cosine),

    %% Restore from ETS (already hydrated by hann_scylla:init)
    Entry    = hann_ets:get_state(entry,     undefined),
    MaxLayer = hann_ets:get_state(max_layer, -1),

    %% Publish ef + dist_fn so search/2 can read them without a call
    hann_ets:put_state([{dist_fn, DistFn}, {ef, Ef}]),

    logger:info("[hann_hnsw] init dim=~w M=~w ef=~w dist=~w "
                "restored entry=~w max_layer=~w",
                [Dim, M, Ef, DistFn, Entry, MaxLayer]),

    {ok, #state{dim       = Dim,
                m         = M,
                m0        = 2 * M,
                ef        = Ef,
                ml        = 1.0 / math:log(M),
                dist_fn   = DistFn,
                entry     = Entry,
                max_layer = MaxLayer}}.

handle_call({add, NodeID, Vector}, _From, State) ->
    case hann_ets:node_exists(NodeID) of
        true  -> {reply, {error, already_exists}, State};
        false ->
            State1 = do_insert(NodeID, Vector, State),
            {reply, ok, State1}
    end;

handle_call({bulk_add, Pairs}, _From, State) ->
    {State1, Count} = lists:foldl(
        fun({ID, Vec}, {St, N}) ->
            case hann_ets:node_exists(ID) of
                true  -> {St, N};
                false -> {do_insert(ID, Vec, St), N + 1}
            end
        end,
        {State, 0},
        Pairs
    ),
    {reply, {ok, Count}, State1};

handle_call(stats, _From, State) ->
    {reply,
     #{node_count => hann_ets:node_count(),
       max_layer  => State#state.max_layer,
       entry      => State#state.entry,
       dim        => State#state.dim,
       m          => State#state.m,
       ef         => State#state.ef,
       dist_fn    => State#state.dist_fn},
     State};

handle_call(_, _, State) ->
    {reply, {error, unknown_call}, State}.

handle_cast(_, State) -> {noreply, State}.
handle_info(_, State) -> {noreply, State}.
terminate(_, _)       -> ok.
code_change(_, S, _)  -> {ok, S}.

%%%-------------------------------------------------------------------
%%% HNSW — Insert  (runs inside gen_server, writes are serialised)
%%%-------------------------------------------------------------------

do_insert(NodeID, Vector, State) ->
    #state{m=M, m0=M0, ef=Ef, ml=ML, dist_fn=DistFn,
           entry=Entry, max_layer=MaxLayer} = State,

    Level = random_level(ML),

    %% Register the new node in ETS BEFORE linking it into the graph.
    %% Concurrent readers may encounter this node ID in a neighbour
    %% list the moment we start writing reverse edges; it must be
    %% resolvable.  Layers map starts empty — readers see an isolated
    %% node briefly, which is harmless for an approximate index.
    hann_ets:insert_node(NodeID, Vector, #{}),

    case Entry of
        %% ── First node ────────────────────────────────────────────────
        undefined ->
            flush_global_state(NodeID, Level, State),
            hann_scylla:async_persist_node(NodeID),
            hann_scylla:persist_global_state(NodeID, Level),
            State#state{entry=NodeID, max_layer=Level};

        %% ── General insert ────────────────────────────────────────────
        _ ->
            %% Phase 1: greedy descent from top layer down to Level+1.
            %%          Only need the single closest node as seed.
            EP1 = case MaxLayer > Level of
                true  -> greedy_descent(Vector, Entry, MaxLayer, Level+1, DistFn);
                false -> Entry
            end,

            Mmax     = fun(0) -> M0; (_) -> M end,
            StartLyr = min(Level, MaxLayer),

            %% Phase 2: beam search + bidirectional wiring at each layer.
            %%          Track which existing nodes had their lists mutated
            %%          so we can enqueue them for ScyllaDB persistence.
            {_, ModifiedSet} = lists:foldl(
                fun(L, {EPAcc, MAcc}) ->
                    W      = search_layer(Vector, EPAcc, Ef, L, DistFn),
                    Nbrs   = select_neighbors_simple(W, Mmax(L)),
                    NbrIDs = [NID || {_, NID} <- Nbrs],

                    %% Forward edges: new node → its neighbours
                    hann_ets:set_layer_neighbors(NodeID, L, NbrIDs),

                    %% Reverse edges: neighbours → new node (with pruning)
                    MAcc1 = lists:foldl(
                        fun(NID, MA) ->
                            Existing = hann_ets:get_layer_neighbors(NID, L),
                            Updated  = [NodeID | Existing],
                            case length(Updated) > Mmax(L) of
                                false ->
                                    hann_ets:set_layer_neighbors(NID, L, Updated);
                                true ->
                                    NVec  = hann_ets:node_vector(NID),
                                    CS    = candidate_set(NVec, Updated, DistFn),
                                    Pruned = select_neighbors_simple(CS, Mmax(L)),
                                    hann_ets:set_layer_neighbors(
                                        NID, L, [I || {_, I} <- Pruned])
                            end,
                            sets:add_element(NID, MA)
                        end,
                        MAcc,
                        NbrIDs
                    ),

                    %% Full W is the entry-point set for the layer below
                    EPNext = [NID || {_, NID} <- gb_sets:to_list(W)],
                    {EPNext, MAcc1}
                end,
                {[EP1], sets:new()},
                lists:seq(StartLyr, 0, -1)
            ),

            NewEntry    = if Level > MaxLayer -> NodeID; true -> Entry end,
            NewMaxLayer = max(Level, MaxLayer),

            flush_global_state(NewEntry, NewMaxLayer, State),

            %% ScyllaDB persistence (async for nodes, sync for global state)
            hann_scylla:async_persist_node(NodeID),
            hann_scylla:async_persist_nodes(sets:to_list(ModifiedSet)),
            hann_scylla:persist_global_state(NewEntry, NewMaxLayer),

            State#state{entry=NewEntry, max_layer=NewMaxLayer}
    end.

%%%-------------------------------------------------------------------
%%% HNSW — Search  (called directly, NOT via gen_server:call)
%%%-------------------------------------------------------------------

do_search(Query, K, Entry, MaxLayer, Ef, DistFn) ->
    %% Greedy descent through layers above 0 → single closest seed
    EP0 = case MaxLayer > 0 of
        true  -> greedy_descent(Query, Entry, MaxLayer, 1, DistFn);
        false -> Entry
    end,
    %% Full beam search at layer 0 with ef ≥ K
    W = search_layer(Query, [EP0], max(Ef, K), 0, DistFn),
    Sorted = lists:sort(gb_sets:to_list(W)),
    lists:sublist(Sorted, K).

%%%-------------------------------------------------------------------
%%% HNSW Core — search_layer
%%%
%%% Returns a gb_set of {Distance, NodeID}, size ≤ Ef.
%%% All node lookups are direct ETS reads — no gen_server involved.
%%%-------------------------------------------------------------------

search_layer(Query, EntryPoints, Ef, Layer, DistFn) ->
    InitPairs = [dist_pair(Query, EP, DistFn) || EP <- EntryPoints],
    C = gb_sets:from_list(InitPairs),   %% candidates  (process closest first)
    W = gb_sets:from_list(InitPairs),   %% result set  (bounded to Ef)
    V = gb_sets:from_list(EntryPoints), %% visited     (node IDs)
    do_search_layer(Query, C, W, Ef, Layer, DistFn, V).

do_search_layer(Query, C, W, Ef, Layer, DistFn, Visited) ->
    case gb_sets:is_empty(C) of
        true -> W;
        false ->
            {CDist, CNode} = gb_sets:smallest(C),
            C1 = gb_sets:delete({CDist, CNode}, C),
            {WWorst, _} = gb_sets:largest(W),
            case CDist > WWorst of
                true  -> W;   %% every remaining candidate is farther than worst result
                false ->
                    Nbrs = hann_ets:get_layer_neighbors(CNode, Layer),
                    {C2, W2, V2} = expand_candidates(
                        Query, Nbrs, C1, W, Ef, DistFn, Visited),
                    do_search_layer(Query, C2, W2, Ef, Layer, DistFn, V2)
            end
    end.

%% @private  Process unvisited neighbours, updating candidate set and result window.
expand_candidates(_, [], C, W, _, _, V) ->
    {C, W, V};
expand_candidates(Query, [E | Rest], C, W, Ef, DistFn, V) ->
    case gb_sets:is_member(E, V) of
        true ->
            expand_candidates(Query, Rest, C, W, Ef, DistFn, V);
        false ->
            V1    = gb_sets:add(E, V),
            EDist = dist(Query, E, DistFn),
            {WWorst, _} = gb_sets:largest(W),
            WSize = gb_sets:size(W),
            {C1, W1} = if
                EDist < WWorst orelse WSize < Ef ->
                    C0  = gb_sets:add({EDist, E}, C),
                    W0  = gb_sets:add({EDist, E}, W),
                    W0t = case gb_sets:size(W0) > Ef of
                        true  ->
                            {_, Trimmed} = gb_sets:take_largest(W0),
                            Trimmed;
                        false -> W0
                    end,
                    {C0, W0t};
                true ->
                    {C, W}
            end,
            expand_candidates(Query, Rest, C1, W1, Ef, DistFn, V1)
    end.

%%%-------------------------------------------------------------------
%%% HNSW Core — greedy descent (top layers → target layer)
%%%-------------------------------------------------------------------

greedy_descent(Vector, EP, FromLayer, ToLayer, DistFn) ->
    lists:foldl(
        fun(L, CurEP) ->
            W = search_layer(Vector, [CurEP], 1, L, DistFn),
            case gb_sets:is_empty(W) of
                true  -> CurEP;
                false -> element(2, gb_sets:smallest(W))
            end
        end,
        EP,
        lists:seq(FromLayer, ToLayer, -1)
    ).

%%%-------------------------------------------------------------------
%%% HNSW Core — neighbor selection (Algorithm 3 — simple greedy)
%%%-------------------------------------------------------------------

select_neighbors_simple(W, M) ->
    Sorted = lists:sort(gb_sets:to_list(W)),
    lists:sublist(Sorted, M).

%%%-------------------------------------------------------------------
%%% Helpers
%%%-------------------------------------------------------------------

%% Build a {Dist, NodeID} candidate gb_set from a raw node-ID list.
candidate_set(RefVec, NodeIDs, DistFn) ->
    gb_sets:from_list([{hann_vector:distance(RefVec,
                         hann_ets:node_vector(X), DistFn), X}
                       || X <- NodeIDs]).

dist(Query, NodeID, DistFn) ->
    hann_vector:distance(Query, hann_ets:node_vector(NodeID), DistFn).

dist_pair(Query, NodeID, DistFn) ->
    {dist(Query, NodeID, DistFn), NodeID}.

%% Publish entry + max_layer into ETS so search/2 sees them without a call.
flush_global_state(Entry, MaxLayer, #state{dist_fn=DistFn, ef=Ef}) ->
    hann_ets:put_state([{entry,     Entry},
                        {max_layer, MaxLayer},
                        {dist_fn,   DistFn},
                        {ef,        Ef}]).

%% Geometric level distribution: P(l) ∝ exp(-l / mL)
random_level(ML) ->
    floor(-math:log(rand:uniform()) * ML).
