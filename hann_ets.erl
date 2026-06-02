%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — ETS Table Owner
%%%
%%% Owns three named, public ETS tables so they outlive any individual
%%% worker crash.  All tables use {read_concurrency, true} so that the
%%% concurrent search path (hann_hnsw:search/2) never contends on an
%%% ERL_NIF lock — multiple schedulers can scan the HNSW graph in
%%% parallel without touching this gen_server at all.
%%%
%%% ── Tables ─────────────────────────────────────────────────────────
%%%
%%%   hann_nodes      {NodeID, Vector, Layers}
%%%                     NodeID  :: non_neg_integer()
%%%                     Vector  :: [float()]           128-d RegionDCL
%%%                     Layers  :: #{Layer => [NodeID]}  adjacency map
%%%
%%%   hann_idx_state  {Key :: atom(), Value}
%%%                     entry     → NodeID | undefined
%%%                     max_layer → integer()
%%%                     dist_fn   → cosine | euclidean
%%%                     ef        → pos_integer()
%%%
%%%   hann_meta       {NodeID, Meta :: map()}
%%%                     h3_index, stop_name, route_ids, region
%%%
%%% ── Access pattern ──────────────────────────────────────────────────
%%%
%%%   Reads  → call module-level helpers directly (no gen_server hop)
%%%   Writes → also direct ets:insert (serialised by hann_hnsw writes)
%%%-------------------------------------------------------------------
-module(hann_ets).
-behaviour(gen_server).

-define(NODES_TAB,  hann_nodes).
-define(STATE_TAB,  hann_idx_state).
-define(META_TAB,   hann_meta).

%% Public helpers — direct ETS ops, zero gen_server overhead
-export([
    insert_node/3,
    node_vector/1,
    node_exists/1,
    get_layer_neighbors/2,
    set_layer_neighbors/3,
    get_node_full/1,
    get_state/2,
    put_state/1,
    get_meta/1,
    put_meta/2,
    node_count/0,
    all_node_ids/0
]).

-export([start_link/0]).
-export([init/1, handle_call/3, handle_cast/2,
         handle_info/2, terminate/2, code_change/3]).

%%%-------------------------------------------------------------------
%%% Public helpers (pure ETS — no gen_server roundtrip)
%%%-------------------------------------------------------------------

%% @doc Write a new node row (Vector + empty Layers map).
-spec insert_node(non_neg_integer(), [float()], map()) -> true.
insert_node(NodeID, Vector, Layers) ->
    ets:insert(?NODES_TAB, {NodeID, Vector, Layers}).

%% @doc Fetch the embedding vector for a node.
-spec node_vector(non_neg_integer()) -> [float()].
node_vector(NodeID) ->
    case ets:lookup(?NODES_TAB, NodeID) of
        [{_, V, _}] -> V;
        []          -> error({node_not_found, NodeID})
    end.

%% @doc Check existence without fetching data.
-spec node_exists(non_neg_integer()) -> boolean().
node_exists(NodeID) ->
    ets:member(?NODES_TAB, NodeID).

%% @doc Retrieve the neighbour list for one layer.
-spec get_layer_neighbors(non_neg_integer(), non_neg_integer()) ->
    [non_neg_integer()].
get_layer_neighbors(NodeID, Layer) ->
    case ets:lookup(?NODES_TAB, NodeID) of
        [{_, _, Ls}] -> maps:get(Layer, Ls, []);
        []           -> []
    end.

%% @doc Overwrite the neighbour list for one layer (atomic row replace).
-spec set_layer_neighbors(non_neg_integer(), non_neg_integer(),
                          [non_neg_integer()]) -> true.
set_layer_neighbors(NodeID, Layer, Neighbors) ->
    case ets:lookup(?NODES_TAB, NodeID) of
        [{_, V, Ls}] ->
            ets:insert(?NODES_TAB, {NodeID, V, maps:put(Layer, Neighbors, Ls)});
        [] ->
            error({node_not_found, NodeID})
    end.

%% @doc Full row read — used by hann_scylla for persistence snapshots.
-spec get_node_full(non_neg_integer()) ->
    {[float()], map()} | undefined.
get_node_full(NodeID) ->
    case ets:lookup(?NODES_TAB, NodeID) of
        [{_, V, Ls}] -> {V, Ls};
        []           -> undefined
    end.

%% @doc Read one HNSW global-state value with a fallback default.
-spec get_state(atom(), term()) -> term().
get_state(Key, Default) ->
    case ets:lookup(?STATE_TAB, Key) of
        [{_, V}] -> V;
        []       -> Default
    end.

%% @doc Batch-write HNSW global-state entries.
%%      KVPairs = [{entry, NodeID}, {max_layer, N}, …]
-spec put_state([{atom(), term()}]) -> true.
put_state(KVPairs) when is_list(KVPairs) ->
    ets:insert(?STATE_TAB, KVPairs).

%% @doc Retrieve metadata map for a node (or undefined).
-spec get_meta(non_neg_integer()) -> map() | undefined.
get_meta(NodeID) ->
    case ets:lookup(?META_TAB, NodeID) of
        [{_, M}] -> M;
        []       -> undefined
    end.

%% @doc Store or merge metadata for a node.
-spec put_meta(non_neg_integer(), map()) -> true.
put_meta(NodeID, Meta) ->
    Existing = case ets:lookup(?META_TAB, NodeID) of
        [{_, M}] -> M;
        []       -> #{}
    end,
    ets:insert(?META_TAB, {NodeID, maps:merge(Existing, Meta)}).

%% @doc Total number of indexed nodes.
-spec node_count() -> non_neg_integer().
node_count() ->
    ets:info(?NODES_TAB, size).

%% @doc All node IDs — used for bulk ScyllaDB sync checks.
-spec all_node_ids() -> [non_neg_integer()].
all_node_ids() ->
    ets:select(?NODES_TAB, [{ {'$1','_','_'}, [], ['$1'] }]).

%%%-------------------------------------------------------------------
%%% gen_server — sole purpose is to own the ETS tables
%%%-------------------------------------------------------------------

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init([]) ->
    %% public + read_concurrency: searches bypass this process entirely.
    %% write_concurrency false: only the hann_hnsw gen_server writes,
    %% so false avoids the extra memory overhead of per-shard locks.
    ets:new(?NODES_TAB, [named_table, public, set,
                         {read_concurrency,  true},
                         {write_concurrency, false}]),
    ets:new(?STATE_TAB, [named_table, public, set,
                         {read_concurrency,  true}]),
    ets:new(?META_TAB,  [named_table, public, set,
                         {read_concurrency,  true}]),
    logger:info("[hann_ets] tables created: ~w ~w ~w",
                [?NODES_TAB, ?STATE_TAB, ?META_TAB]),
    {ok, #{}}.

handle_call(_, _, S) -> {reply, ok, S}.
handle_cast(_, S)    -> {noreply, S}.
handle_info(_, S)    -> {noreply, S}.
terminate(_, _)      -> ok.
code_change(_, S, _) -> {ok, S}.
