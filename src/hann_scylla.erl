%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — ScyllaDB persistence via marina
%%%
%%% ── marina API (this version) ──────────────────────────────────────
%%%
%%%   marina:query/2         synchronous query
%%%   marina:async_query/2   returns Ref; marina:receive_response/1 to wait
%%%
%%%   Verified from: marina:module_info(exports) in the shell.
%%%
%%%   Adapter: marina_call/3 — single function to update if the Request
%%%   format differs from #{query, values, consistency}.
%%%   Check: _build/default/lib/marina/src/marina.erl for exact spec.
%%%
%%% ── Startup ─────────────────────────────────────────────────────────
%%%
%%%   init/1 returns immediately (ready=false) and sends self a
%%%   `connect` message.  hann_scylla never crashes the supervisor
%%%   on boot regardless of ScyllaDB availability.
%%%
%%%   Retries every 5 s until ScyllaDB is reachable.
%%%   All writes while not ready are silently dropped (ETS is durable).
%%%
%%% ── Write path ──────────────────────────────────────────────────────
%%%
%%%   Async node/meta writes are spawned — they never block the
%%%   gen_server mailbox.  Global state (entry + max_layer) is written
%%%   synchronously because hann_hnsw:add/2 must not return until the
%%%   graph cursor is durable.
%%%-------------------------------------------------------------------
-module(hann_scylla).
-behaviour(gen_server).

-export([start_link/0,
         async_persist_node/1,
         async_persist_nodes/1,
         async_persist_meta/2,
         persist_global_state/2,
         is_ready/0]).

-export([init/1, handle_call/3, handle_cast/2,
         handle_info/2, terminate/2, code_change/3]).

-define(RETRY_MS, 5000).

%%%-------------------------------------------------------------------
%%% CQL
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
    ready :: boolean()
}).

%%%-------------------------------------------------------------------
%%% Public API
%%%-------------------------------------------------------------------

start_link() ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

-spec async_persist_node(non_neg_integer()) -> ok.
async_persist_node(NodeID) ->
    gen_server:cast(?MODULE, {persist_node, NodeID}).

-spec async_persist_nodes([non_neg_integer()]) -> ok.
async_persist_nodes([])  -> ok;
async_persist_nodes(IDs) -> gen_server:cast(?MODULE, {persist_nodes, IDs}).

-spec async_persist_meta(non_neg_integer(), map()) -> ok.
async_persist_meta(NodeID, Meta) ->
    gen_server:cast(?MODULE, {persist_meta, NodeID, Meta}).

-spec persist_global_state(non_neg_integer() | undefined, integer()) -> ok.
persist_global_state(Entry, MaxLayer) ->
    gen_server:call(?MODULE, {persist_state, Entry, MaxLayer}, 10_000).

-spec is_ready() -> boolean().
is_ready() ->
    gen_server:call(?MODULE, is_ready, 5_000).

%%%-------------------------------------------------------------------
%%% gen_server callbacks
%%%-------------------------------------------------------------------

init([]) ->
    self() ! connect,
    logger:info("[hann_scylla] started — awaiting ScyllaDB"),
    {ok, #state{ready = false}}.

handle_call(is_ready, _From, State) ->
    {reply, State#state.ready, State};

handle_call({persist_state, _E, _M}, _From, #state{ready = false} = S) ->
    {reply, ok, S};

handle_call({persist_state, Entry, MaxLayer}, _From, State) ->
    marina_call(?Q_UPSERT_STATE,
                [?STATE_KEY, coerce_entry(Entry), MaxLayer], quorum),
    {reply, ok, State};

handle_call(_, _, State) ->
    {reply, ok, State}.

handle_cast(_, #state{ready = false} = State) ->
    {noreply, State};

handle_cast({persist_node, NodeID}, State) ->
    spawn_persist_node(NodeID),
    {noreply, State};

handle_cast({persist_nodes, NodeIDs}, State) ->
    lists:foreach(fun spawn_persist_node/1, NodeIDs),
    {noreply, State};

handle_cast({persist_meta, NodeID, Meta}, State) ->
    spawn_persist_meta(NodeID, Meta),
    {noreply, State};

handle_cast(_, State) ->
    {noreply, State}.

handle_info(connect, State) ->
    case do_connect() of
        ok ->
            logger:info("[hann_scylla] connected — schema ensured, ETS hydrated"),
            {noreply, State#state{ready = true}};
        {error, Reason} ->
            logger:warning("[hann_scylla] connect failed: ~p — retry in ~wms",
                           [Reason, ?RETRY_MS]),
            erlang:send_after(?RETRY_MS, self(), connect),
            {noreply, State#state{ready = false}}
    end;

handle_info(_, State) ->
    {noreply, State}.

terminate(_, _)      -> ok.
code_change(_, S, _) -> {ok, S}.

%%%-------------------------------------------------------------------
%%% Connect + schema + hydration
%%%-------------------------------------------------------------------

do_connect() ->
    try
        ensure_schema(),
        load_all(),
        ok
    catch
        _:Reason -> {error, Reason}
    end.

ensure_schema() ->
    lists:foreach(fun run_ddl/1,
                  [?Q_CREATE_KS, ?Q_CREATE_NODES,
                   ?Q_CREATE_META, ?Q_CREATE_STATE]).

run_ddl(Stmt) ->
    case marina_call(Stmt, [], one) of
        {ok, _}    -> ok;
        {error, R} -> logger:debug("[hann_scylla] DDL: ~p", [R])
    end.

load_all() ->
    load_nodes(),
    load_meta(),
    load_state().

load_nodes() ->
    case marina_call(?Q_SELECT_ALL_NODES, [], one) of
        {ok, []} ->
            logger:info("[hann_scylla] nodes table empty");
        {ok, [{_KS, _Cols, Rows}]} ->
            lists:foreach(fun hydrate_node/1, Rows),
            logger:info("[hann_scylla] hydrated ~w nodes", [length(Rows)]);
        {error, R} ->
            logger:warning("[hann_scylla] load_nodes: ~p", [R])
    end.

hydrate_node({NodeID, VecBin, LayersBin}) ->
    hann_ets:insert_node(NodeID, decode_vector(VecBin),
                         binary_to_term(LayersBin, [safe])).

load_meta() ->
    case marina_call(?Q_SELECT_ALL_META, [], one) of
        {ok, [{_KS, _Cols, Rows}]} ->
            lists:foreach(
                fun({NodeID, H3, Stop, RoutesBin, Region}) ->
                    hann_ets:put_meta(NodeID,
                        #{h3_index  => null_or(H3),
                          stop_name => null_or(Stop),
                          route_ids => binary_to_term(RoutesBin, [safe]),
                          region    => null_or(Region)})
                end, Rows);
        _ -> ok
    end.

load_state() ->
    case marina_call(?Q_SELECT_STATE, [?STATE_KEY], one) of
        {ok, [{_KS, _Cols, [{Entry, MaxLayer}]}]} ->
            hann_ets:put_state([{entry, Entry}, {max_layer, MaxLayer}]),
            logger:info("[hann_scylla] state restored entry=~w max_layer=~w",
                        [Entry, MaxLayer]);
        _ ->
            hann_ets:put_state([{entry, undefined}, {max_layer, -1}]),
            logger:info("[hann_scylla] no prior state — fresh start")
    end.

%%%-------------------------------------------------------------------
%%% Spawned async persistence
%%%-------------------------------------------------------------------

spawn_persist_node(NodeID) ->
    case hann_ets:get_node_full(NodeID) of
        undefined -> ok;
        {Vector, Layers} ->
            VecBin    = encode_vector(Vector),
            LayersBin = term_to_binary(Layers),
            spawn(fun() ->
                case marina_call(?Q_UPSERT_NODE,
                                 [NodeID, VecBin, LayersBin], one) of
                    {ok, _}    -> ok;
                    {error, R} ->
                        logger:warning("[hann_scylla] node ~w persist: ~p",
                                       [NodeID, R])
                end
            end)
    end,
    ok.

spawn_persist_meta(NodeID, Meta) ->
    H3        = maps:get(h3_index,  Meta, null),
    Stop      = maps:get(stop_name, Meta, null),
    Routes    = maps:get(route_ids, Meta, []),
    Region    = maps:get(region,    Meta, null),
    RoutesBin = term_to_binary(Routes),
    spawn(fun() ->
        case marina_call(?Q_UPSERT_META,
                         [NodeID, H3, Stop, RoutesBin, Region], one) of
            {ok, _}    -> ok;
            {error, R} ->
                logger:warning("[hann_scylla] meta ~w persist: ~p", [NodeID, R])
        end
    end),
    ok.

%%%-------------------------------------------------------------------
%%% marina adapter
%%%
%%% marina:query/2 is the API in this version.
%%% Adjust the Request format below if you get a badarg or function_clause
%%% error — check _build/default/lib/marina/src/marina.erl for the spec.
%%%
%%% Common formats to try if #{query,values,consistency} doesn't work:
%%%   {Query, Values, Consistency}
%%%   marina:query(Query, #{values => V, consistency => C})
%%%-------------------------------------------------------------------

marina_call(Query, Values, Consistency) ->
    Request = #{
        query       => Query,
        values      => Values,
        consistency => Consistency
    },
    marina:query(Request, #{}).

%%%-------------------------------------------------------------------
%%% Codec
%%%-------------------------------------------------------------------

encode_vector(Vec) ->
    << <<X:32/float-little>> || X <- Vec >>.

decode_vector(Bin) ->
    [X || <<X:32/float-little>> <= Bin].

coerce_entry(undefined) -> 0;
coerce_entry(N)         -> N.

null_or(null) -> null;
null_or(<<>>) -> null;
null_or(V)    -> V.
