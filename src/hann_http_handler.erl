%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — Cowboy HTTP REST Handler
%%%
%%% Routes (set in himap_hann_app.erl):
%%%
%%%   GET  /health                 → 200 {status: "ok", …}
%%%
%%%   POST /v1/node                → add one node
%%%        Body: {id: int, vector: [float, …]}
%%%
%%%   POST /v1/index/bulk          → add many nodes
%%%        Body: {nodes: [{id, vector}, …]}
%%%
%%%   POST /v1/search              → raw HNSW kNN search
%%%        Body: {vector: [float, …], k: int}
%%%
%%%   POST /v1/grg/query           → Cloverleaf GRG query
%%%        Body: {regiondcl_embedding: [float, …], k: int}
%%%
%%%   POST /v1/grg/meta            → register node metadata
%%%        Body: {node_id: int, meta: {h3_index, stop_name, …}}
%%%-------------------------------------------------------------------
-module(hann_http_handler).
-behaviour(cowboy_handler).

-export([init/2]).

%%%-------------------------------------------------------------------
%%% Cowboy entry point
%%%-------------------------------------------------------------------

init(Req0, State = #{action := Action}) ->
    {Code, Body} = dispatch(Action, Req0),
    Req1 = cowboy_req:reply(
        Code,
        #{<<"content-type">> => <<"application/json">>},
        jiffy:encode(Body),
        Req0
    ),
    {ok, Req1, State}.

%%%-------------------------------------------------------------------
%%% Route dispatchers
%%%-------------------------------------------------------------------

dispatch(health, _Req) ->
    Stats = hann_hnsw:stats(),
    {200, #{status => <<"ok">>,
            node_count => maps:get(node_count, Stats, 0),
            max_layer  => maps:get(max_layer,  Stats, -1)}};

dispatch(add_node, Req) ->
    case read_json(Req) of
        {error, Reason} ->
            bad_request(Reason);
        {ok, #{<<"id">> := NodeID, <<"vector">> := VecRaw}} ->
            Vector = to_float_list(VecRaw),
            case hann_hnsw:add(NodeID, Vector) of
                ok             -> {201, #{status => <<"inserted">>, id => NodeID}};
                {error, Reason} -> bad_request(Reason)
            end;
        _ ->
            bad_request(<<"missing fields: id, vector">>)
    end;

dispatch(bulk_add, Req) ->
    case read_json(Req) of
        {error, Reason} ->
            bad_request(Reason);
        {ok, #{<<"nodes">> := Nodes}} when is_list(Nodes) ->
            Pairs = [{maps:get(<<"id">>, N), to_float_list(maps:get(<<"vector">>, N))}
                     || N <- Nodes],
            {ok, Count} = hann_hnsw:bulk_add(Pairs),
            {200, #{status => <<"ok">>, inserted => Count}};
        _ ->
            bad_request(<<"expected {nodes: [{id, vector}]}">>)
    end;

dispatch(search, Req) ->
    case read_json(Req) of
        {error, Reason} ->
            bad_request(Reason);
        {ok, #{<<"vector">> := VecRaw, <<"k">> := K}} ->
            Vector  = to_float_list(VecRaw),
            Results = hann_hnsw:search(Vector, K),
            Payload = [#{node_id => NID, distance => D} || {D, NID} <- Results],
            {200, #{results => Payload, count => length(Payload)}};
        _ ->
            bad_request(<<"missing fields: vector, k">>)
    end;

dispatch(grg_query, Req) ->
    case read_json(Req) of
        {error, Reason} ->
            bad_request(Reason);
        {ok, Parsed} ->
            Embedding = to_float_list(maps:get(<<"regiondcl_embedding">>, Parsed, [])),
            K         = maps:get(<<"k">>, Parsed, 5),
            case Embedding of
                [] ->
                    bad_request(<<"regiondcl_embedding must be a non-empty float array">>);
                _ ->
                    Neighbors = hann_grg:query_neighbors(Embedding, K),
                    Payload   = [serialize_neighbor(N) || N <- Neighbors],
                    {200, #{grg_neighbors => Payload, count => length(Payload)}}
            end;
        _ ->
            bad_request(<<"missing field: regiondcl_embedding">>)
    end;

dispatch(register_meta, Req) ->
    case read_json(Req) of
        {ok, #{<<"node_id">> := NodeID, <<"meta">> := MetaRaw}} ->
            Meta = atomise_meta(MetaRaw),
            hann_grg:register_node_meta(NodeID, Meta),
            {200, #{status => <<"registered">>, node_id => NodeID}};
        {error, Reason} ->
            bad_request(Reason);
        _ ->
            bad_request(<<"missing fields: node_id, meta">>)
    end;

dispatch(_Unknown, _Req) ->
    {404, #{error => <<"not_found">>}}.

%%%-------------------------------------------------------------------
%%% Internal helpers
%%%-------------------------------------------------------------------

read_json(Req) ->
    {ok, Body, _Req1} = cowboy_req:read_body(Req),
    try
        Parsed = jiffy:decode(Body, [return_maps]),
        {ok, Parsed}
    catch
        _:_ -> {error, <<"invalid JSON">>}
    end.

bad_request(Reason) ->
    {400, #{error => ensure_binary(Reason)}}.

to_float_list(List) when is_list(List) ->
    [float(X) || X <- List];
to_float_list(_) ->
    [].

%% Rename binary keys to atoms for internal use
atomise_meta(Map) when is_map(Map) ->
    maps:fold(
        fun(K, V, Acc) ->
            AtomKey = binary_to_existing_atom_safe(K),
            maps:put(AtomKey, V, Acc)
        end,
        #{},
        Map
    );
atomise_meta(_) -> #{}.

binary_to_existing_atom_safe(B) ->
    try binary_to_existing_atom(B, utf8)
    catch _:_ -> B   %% unknown key — keep as binary
    end.

serialize_neighbor(N) ->
    #{node_id    => maps:get(node_id,    N, null),
      distance   => maps:get(distance,   N, null),
      similarity => maps:get(similarity, N, null),
      rank       => maps:get(rank,       N, null),
      h3_index   => maps:get(h3_index,   N, null),
      stop_name  => maps:get(stop_name,  N, null),
      route_ids  => maps:get(route_ids,  N, []),
      region     => maps:get(region,     N, null)}.

ensure_binary(B) when is_binary(B) -> B;
ensure_binary(A) when is_atom(A)   -> atom_to_binary(A, utf8);
ensure_binary(T)                   -> list_to_binary(io_lib:format("~p", [T])).
