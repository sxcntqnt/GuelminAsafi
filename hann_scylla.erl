%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — ScyllaDB Persistence via marina
%%%
%%% Wraps lpgauth/marina (pure-Erlang native CQL protocol client) to
%%% provide durable storage for the HNSW graph and GRG metadata.
%%%
%%% ── Startup sequence ────────────────────────────────────────────────
%%%
%%%   1. ensure_schema/0  — DDL (CREATE IF NOT EXISTS), tolerates
%%%                         "already exists" errors from ScyllaDB.
%%%   2. load_all/0       — hydrates hann_ets tables from ScyllaDB so
%%%                         hann_hnsw:init/1 can restore its state
%%%                         without a gen_server roundtrip.
%%%
%%% ── Write path ──────────────────────────────────────────────────────
%%%
%%%   Writes are fire-and-forget (marina async_query).
%%%   marina_response messages are handled in handle_info/2 for logging.
%%%   persist_global_state/2 is synchronous because hann_hnsw needs
%%%   the entry-point durable before acknowledging an add/2 call.
%%%
%%% ── Encoding ────────────────────────────────────────────────────────
%%%
%%%   vector  → packed float32-LE binary  (128 × 4 = 512 B)
%%%   layers  → term_to_binary/1          (#{Layer => [NodeID]})
%%%   meta    → individual CQL columns + term_to_binary for route_ids
%%%-------------------------------------------------------------------
-module(hann_scylla).
-behaviour(gen_server).

-export([start_link/0,
         async_persist_node/1,
         async_persist_nodes/1,
         async_persist_meta/2,
         persist_global_state/2]).

-export([init/1, handle_call/3, handle_cast/2,
         handle_info/2, terminate/2, code_change/3]).

%%%-------------------------------------------------------------------
%%% CQL statements
%%%-------------------------------------------------------------------

-define(Q_CREATE_KS,
    <<"CREATE KEYSPACE IF NOT EXISTS himap "
      "WITH replication = {'class': 'SimpleStrategy', "
      "'replication_factor': 1} AND durable_writes = true">>).

-define(Q_CREATE_NODES,
    <<"CREATE TABLE IF NOT EXISTS himap.nodes "
      "(node_id bigint PRIMARY KEY, vector blob, layers blob)">>).

-define(Q_CREATE_META,
    <<"CREATE TABLE IF NOT EXISTS himap.node_meta "
      "(node_id bigint PRIMARY KEY, h3_index text, "
      "stop_name text, route_ids blob, region text)">>).

-define(Q_CREATE_STATE,
    <<"CREATE TABLE IF NOT EXISTS himap.hnsw_state "
      "(key text PRIMARY KEY, entry bigint, max_layer int)">>).

-define(Q_UPSERT_NODE,
    <<"INSERT INTO himap.nodes (node_id, vector, layers) VALUES (?, ?, ?)">>).

-define(Q_SELECT_ALL_NODES,
    <<"SELECT node_id, vector, layers FROM himap.nodes">>).

-define(Q_UPSERT_META,
    <<"INSERT INTO himap.node_meta "
      "(node_id, h3_index, stop_name, route_ids, region) "
      "VALUES (?, ?, ?, ?, ?)">>).

-define(Q_SELECT_ALL_META,
    <<"SELECT node_id, h3_index, stop_name, route_ids, region "
      "FROM himap.node_meta">>).

-define(Q_UPSERT_STATE,
    <<"INSERT INTO himap.hnsw_state (key, entry, max_layer) VALUES (?, ?, ?)">>).

-define(Q_SELECT_STATE,
    <<"SELECT entry, max_layer FROM himap.hnsw_state WHERE key = ?">>).

-define(STATE_KEY, <<"hnsw">>).

%%%-------------------------------------------------------------------
%%% State
%%%-------------------------------------------------------------------

-record(state, {
    %% In-flight marina async refs → context atom for error attribution
    pending :: #{reference() => term()}
}).

%%%-------------------------------------------------------------------
%%% Public API
%%%-------------------------------------------------------------------

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

%% @doc Enqueue an async ScyllaDB upsert for a single node.
%%      Reads Vector + Layers from hann_ets at cast-processing time
%%      so the snapshot is consistent with what landed in ETS.
-spec async_persist_node(non_neg_integer()) -> ok.
async_persist_node(NodeID) ->
    gen_server:cast(?MODULE, {persist_node, NodeID}).

%% @doc Enqueue async upserts for a list of nodes (post-insert fanout).
-spec async_persist_nodes([non_neg_integer()]) -> ok.
async_persist_nodes([])      -> ok;
async_persist_nodes(NodeIDs) ->
    gen_server:cast(?MODULE, {persist_nodes, NodeIDs}).

%% @doc Enqueue async upsert for node metadata.
-spec async_persist_meta(non_neg_integer(), map()) -> ok.
async_persist_meta(NodeID, Meta) ->
    gen_server:cast(?MODULE, {persist_meta, NodeID, Meta}).

%% @doc Synchronous write of HNSW entry-point + max-layer.
%%      Called by hann_hnsw after every successful insert so a clean
%%      crash-restart can resume without rescanning the whole graph.
-spec persist_global_state(non_neg_integer() | undefined, integer()) -> ok.
persist_global_state(Entry, MaxLayer) ->
    gen_server:call(?MODULE, {persist_state, Entry, MaxLayer}, 10_000).

%%%-------------------------------------------------------------------
%%% gen_server callbacks
%%%-------------------------------------------------------------------

init([]) ->
    logger:info("[hann_scylla] initialising — ensuring schema then loading ETS"),
    ensure_schema(),
    load_all(),
    {ok, #state{pending = #{}}}.

%%% ── Synchronous call: global state persist ─────────────────────────

handle_call({persist_state, Entry, MaxLayer}, _From, State) ->
    case marina:query(?Q_UPSERT_STATE,
                      [?STATE_KEY, coerce_entry(Entry), MaxLayer],
                      quorum, []) of
        {ok, _}    -> ok;
        {error, R} ->
            logger:warning("[hann_scylla] persist_state failed: ~p", [R])
    end,
    {reply, ok, State};

handle_call(_, _, State) ->
    {reply, ok, State}.

%%% ── Async cast handlers ─────────────────────────────────────────────

handle_cast({persist_node, NodeID}, State) ->
    {noreply, fire_node(NodeID, State)};

handle_cast({persist_nodes, NodeIDs}, State) ->
    State1 = lists:foldl(fun fire_node/2, State, NodeIDs),
    {noreply, State1};

handle_cast({persist_meta, NodeID, Meta}, State) ->
    {noreply, fire_meta(NodeID, Meta, State)};

handle_cast(_, State) ->
    {noreply, State}.

%%% ── marina async responses ──────────────────────────────────────────

handle_info({marina_response, Ref, {ok, _}}, State) ->
    {noreply, State#state{pending = maps:remove(Ref, State#state.pending)}};

handle_info({marina_response, Ref, {error, Reason}}, State) ->
    Pending = State#state.pending,
    Ctx     = maps:get(Ref, Pending, unknown),
    logger:warning("[hann_scylla] async write failed ctx=~w reason=~p",
                   [Ctx, Reason]),
    {noreply, State#state{pending = maps:remove(Ref, Pending)}};

handle_info(_, State) ->
    {noreply, State}.

terminate(_, _) -> ok.
code_change(_, State, _) -> {ok, State}.

%%%-------------------------------------------------------------------
%%% Internal — schema DDL
%%%-------------------------------------------------------------------

ensure_schema() ->
    lists:foreach(fun run_ddl/1, [
        ?Q_CREATE_KS,
        ?Q_CREATE_NODES,
        ?Q_CREATE_META,
        ?Q_CREATE_STATE
    ]).

run_ddl(Stmt) ->
    case marina:query(Stmt, [], one, []) of
        {ok, _} ->
            ok;
        {error, Reason} ->
            %% "already exists" comes back as an error too; log and continue
            logger:debug("[hann_scylla] DDL notice: ~p", [Reason])
    end.

%%%-------------------------------------------------------------------
%%% Internal — startup ETS hydration
%%%-------------------------------------------------------------------

load_all() ->
    load_nodes(),
    load_meta(),
    load_state().

load_nodes() ->
    case marina:query(?Q_SELECT_ALL_NODES, [], one, []) of
        {ok, []} ->
            logger:info("[hann_scylla] nodes table empty — fresh index");
        {ok, [{_KS, _Cols, Rows}]} ->
            lists:foreach(fun hydrate_node/1, Rows),
            logger:info("[hann_scylla] hydrated ~w nodes into ETS", [length(Rows)]);
        {error, R} ->
            logger:warning("[hann_scylla] load_nodes failed: ~p — starting empty", [R])
    end.

hydrate_node({NodeID, VecBin, LayersBin}) ->
    Vector = decode_vector(VecBin),
    Layers = binary_to_term(LayersBin, [safe]),
    hann_ets:insert_node(NodeID, Vector, Layers).

load_meta() ->
    case marina:query(?Q_SELECT_ALL_META, [], one, []) of
        {ok, [{_KS, _Cols, Rows}]} ->
            lists:foreach(
                fun({NodeID, H3, Stop, RoutesBin, Region}) ->
                    Meta = #{h3_index  => null_or(H3),
                             stop_name => null_or(Stop),
                             route_ids => binary_to_term(RoutesBin, [safe]),
                             region    => null_or(Region)},
                    hann_ets:put_meta(NodeID, Meta)
                end,
                Rows
            );
        _ ->
            ok
    end.

load_state() ->
    case marina:query(?Q_SELECT_STATE, [?STATE_KEY], one, []) of
        {ok, [{_KS, _Cols, [{Entry, MaxLayer}]}]} ->
            hann_ets:put_state([{entry, Entry}, {max_layer, MaxLayer}]),
            logger:info("[hann_scylla] HNSW state restored "
                        "entry=~w max_layer=~w", [Entry, MaxLayer]);
        _ ->
            hann_ets:put_state([{entry, undefined}, {max_layer, -1}]),
            logger:info("[hann_scylla] no prior HNSW state — beginning fresh")
    end.

%%%-------------------------------------------------------------------
%%% Internal — async fire helpers
%%%-------------------------------------------------------------------

fire_node(NodeID, State) ->
    case hann_ets:get_node_full(NodeID) of
        undefined ->
            State;
        {Vector, Layers} ->
            VecBin    = encode_vector(Vector),
            LayersBin = term_to_binary(Layers),
            Ref = marina:async_query(
                      ?Q_UPSERT_NODE,
                      [NodeID, VecBin, LayersBin],
                      one, [], self()
                  ),
            track(Ref, {node, NodeID}, State)
    end.

fire_meta(NodeID, Meta, State) ->
    H3        = maps:get(h3_index,  Meta, null),
    Stop      = maps:get(stop_name, Meta, null),
    Routes    = maps:get(route_ids, Meta, []),
    Region    = maps:get(region,    Meta, null),
    RoutesBin = term_to_binary(Routes),
    Ref = marina:async_query(
              ?Q_UPSERT_META,
              [NodeID, H3, Stop, RoutesBin, Region],
              one, [], self()
          ),
    track(Ref, {meta, NodeID}, State).

track(Ref, Ctx, State) ->
    State#state{pending = maps:put(Ref, Ctx, State#state.pending)}.

%%%-------------------------------------------------------------------
%%% Internal — codec
%%%-------------------------------------------------------------------

%% Float32-LE packed binary — compact (4 bytes/dim vs 8 for float64).
encode_vector(Vec) ->
    << <<X:32/float-little>> || X <- Vec >>.

decode_vector(Bin) ->
    [X || <<X:32/float-little>> <= Bin].

coerce_entry(undefined) -> 0;   %% ScyllaDB bigint cannot be NULL in this schema
coerce_entry(N)         -> N.

null_or(null) -> null;
null_or(<<>>) -> null;
null_or(V)    -> V.
