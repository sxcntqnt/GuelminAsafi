%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — Cloverleaf GRG Interface
%%%
%%% Translates raw HNSW nearest-neighbour results into the typed
%%% grg_neighbor structures that the Semantic Cognition Layer's input
%%% contract expects.
%%%
%%% Metadata (H3 index, stop names, route IDs) is read from the
%%% hann_meta ETS table managed by hann_ets — zero gen_server hops
%%% on the hot query path.
%%%
%%% ── GRG Neighbor map ────────────────────────────────────────────────
%%%
%%%   #{node_id    => integer(),
%%%     distance   => float(),      cosine distance ∈ [0, 2]
%%%     similarity => float(),      1 – distance/2  ∈ [0, 1]
%%%     rank       => integer(),    1 = closest
%%%     h3_index   => binary() | null,
%%%     stop_name  => binary() | null,
%%%     route_ids  => [integer()],
%%%     region     => binary() | null}
%%%-------------------------------------------------------------------
-module(hann_grg).
-behaviour(gen_server).

-export([start_link/1,
         query_neighbors/2,
         query_neighbors/1,
         register_node_meta/2,
         get_node_meta/1]).

-export([init/1, handle_call/3, handle_cast/2,
         handle_info/2, terminate/2, code_change/3]).

-define(DEFAULT_K, 5).

%%%-------------------------------------------------------------------
%%% Public API
%%%-------------------------------------------------------------------

start_link(Opts) ->
    gen_server:start_link({local, ?MODULE}, ?MODULE, Opts, []).

%% @doc Primary Cognition Layer endpoint.
%%      Queries HNSW (concurrent read, no gen_server) then enriches
%%      results with metadata from hann_meta ETS.
-spec query_neighbors([float()], pos_integer()) -> [map()].
query_neighbors(RegionDCLEmbedding, K) ->
    %% hann_hnsw:search/2 is a direct ETS read — no gen_server call
    Raw = hann_hnsw:search(RegionDCLEmbedding, K),
    enrich(Raw).

-spec query_neighbors([float()]) -> [map()].
query_neighbors(RegionDCLEmbedding) ->
    query_neighbors(RegionDCLEmbedding, ?DEFAULT_K).

%% @doc Register spatial metadata for a node.
%%      Written to hann_meta ETS immediately and persisted to
%%      ScyllaDB asynchronously.
-spec register_node_meta(non_neg_integer(), map()) -> ok.
register_node_meta(NodeID, Meta) ->
    hann_ets:put_meta(NodeID, Meta),
    hann_scylla:async_persist_meta(NodeID, Meta),
    ok.

%% @doc Retrieve stored metadata for a node.
-spec get_node_meta(non_neg_integer()) -> map() | undefined.
get_node_meta(NodeID) ->
    hann_ets:get_meta(NodeID).

%%%-------------------------------------------------------------------
%%% gen_server callbacks — minimal; real work is done in public fns
%%%-------------------------------------------------------------------

init(_Opts) ->
    logger:info("[hann_grg] Cloverleaf GRG interface started"),
    {ok, #{}}.

handle_call(_, _, S) -> {reply, ok, S}.
handle_cast(_, S)    -> {noreply, S}.
handle_info(_, S)    -> {noreply, S}.
terminate(_, _)      -> ok.
code_change(_, S, _) -> {ok, S}.

%%%-------------------------------------------------------------------
%%% Internal — result enrichment
%%%-------------------------------------------------------------------

%% Convert [{Distance, NodeID}] → [grg_neighbor map], attach metadata + rank.
enrich(RawResults) ->
    {Neighbors, _} = lists:foldl(
        fun({Distance, NodeID}, {Acc, Rank}) ->
            Similarity = erlang:max(0.0, 1.0 - Distance / 2.0),
            Meta       = hann_ets:get_meta(NodeID),
            Neighbor   = #{
                node_id    => NodeID,
                distance   => Distance,
                similarity => Similarity,
                rank       => Rank,
                h3_index   => meta_field(Meta, h3_index),
                stop_name  => meta_field(Meta, stop_name),
                route_ids  => meta_field(Meta, route_ids, []),
                region     => meta_field(Meta, region)
            },
            {[Neighbor | Acc], Rank + 1}
        end,
        {[], 1},
        RawResults
    ),
    lists:reverse(Neighbors).

meta_field(undefined, _Key)          -> null;
meta_field(undefined, _Key, Default) -> Default;
meta_field(Meta, Key)                -> maps:get(Key, Meta, null).
meta_field(Meta, Key, Default)       -> maps:get(Key, Meta, Default).
