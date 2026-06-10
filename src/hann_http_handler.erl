%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — Cowboy HTTP Handler
%%%
%%% Routes:
%%%   GET  /health
%%%   POST /v1/node
%%%   POST /v1/index/bulk
%%%   POST /v1/search
%%%   POST /v1/grg/query
%%%   POST /v1/grg/meta
%%%   POST /v1/stream/hypntyz
%%%   POST /v1/stream/shek3m
%%%   POST /v1/stream/region
%%%   GET  /v1/stream/results
%%%   GET  /v1/stream/counters
%%%-------------------------------------------------------------------
-module(hann_http_handler).
-behaviour(cowboy_handler).

-export([init/2]).

init(Req0, State = #{action := Action}) ->
    {Code, Body} = dispatch(Action, Req0),
    Req1 = cowboy_req:reply(
        Code,
        #{<<"content-type">> => <<"application/json">>},
        jiffy:encode(Body),
        Req0
    ),
    {ok, Req1, State}.

%%====================================================================
%% HNSW + GRG routes
%%====================================================================

dispatch(health, _Req) ->
    Stats = hann_hnsw:stats(),
    {200, #{status     => <<"ok">>,
            node_count => maps:get(node_count, Stats, 0),
            max_layer  => maps:get(max_layer,  Stats, -1)}};

dispatch(add_node, Req) ->
    case read_json(Req) of
        {ok, #{<<"id">> := ID, <<"vector">> := V}} ->
            case hann_hnsw:add(ID, to_float_list(V)) of
                ok              -> {201, #{status => <<"inserted">>, id => ID}};
                {error, Reason} -> bad_request(Reason)
            end;
        {error, R} -> bad_request(R);
        _          -> bad_request(<<"missing fields: id, vector">>)
    end;

dispatch(bulk_add, Req) ->
    case read_json(Req) of
        {ok, #{<<"nodes">> := Nodes}} when is_list(Nodes) ->
            Pairs = [{maps:get(<<"id">>, N),
                      to_float_list(maps:get(<<"vector">>, N))}
                     || N <- Nodes],
            {ok, Count} = hann_hnsw:bulk_add(Pairs),
            {200, #{status => <<"ok">>, inserted => Count}};
        {error, R} -> bad_request(R);
        _          -> bad_request(<<"expected {nodes:[{id,vector}]}">>)
    end;

dispatch(search, Req) ->
    case read_json(Req) of
        {ok, #{<<"vector">> := V, <<"k">> := K}} ->
            Results = hann_hnsw:search(to_float_list(V), K),
            Payload = [#{node_id => NID, distance => D} || {D, NID} <- Results],
            {200, #{results => Payload, count => length(Payload)}};
        {error, R} -> bad_request(R);
        _          -> bad_request(<<"missing fields: vector, k">>)
    end;

dispatch(grg_query, Req) ->
    case read_json(Req) of
        {ok, Parsed} ->
            Emb = to_float_list(maps:get(<<"regiondcl_embedding">>, Parsed, [])),
            K   = maps:get(<<"k">>, Parsed, 5),
            case Emb of
                [] -> bad_request(<<"regiondcl_embedding required">>);
                _  ->
                    Neighbors = hann_grg:query_neighbors(Emb, K),
                    Payload   = [serialize_neighbor(N) || N <- Neighbors],
                    {200, #{grg_neighbors => Payload, count => length(Payload)}}
            end;
        {error, R} -> bad_request(R)
    end;

dispatch(register_meta, Req) ->
    case read_json(Req) of
        {ok, #{<<"node_id">> := ID, <<"meta">> := MetaRaw}} ->
            Meta = atomise_meta(MetaRaw),
            hann_grg:register_node_meta(ID, Meta),
            {200, #{status => <<"registered">>, node_id => ID}};
        {error, R} -> bad_request(R);
        _          -> bad_request(<<"missing fields: node_id, meta">>)
    end;

%%====================================================================
%% LocalStreamBuffer routes
%%====================================================================

dispatch(stream_hypntyz, Req) ->
    case read_json(Req) of
        {ok, Raw} ->
            Frame = parse_hypntyz(Raw),
            local_stream_buffer:ingest_hypntyz(hann_lsb, Frame),
            {200, #{status => <<"accepted">>}};
        {error, R} -> bad_request(R)
    end;

dispatch(stream_shek3m, Req) ->
    case read_json(Req) of
        {ok, Raw} ->
            Frame = parse_shek3m(Raw),
            local_stream_buffer:ingest_shek3m(hann_lsb, Frame),
            {200, #{status => <<"accepted">>}};
        {error, R} -> bad_request(R)
    end;

dispatch(stream_region, Req) ->
    case read_json(Req) of
        {ok, Raw} ->
            RE = parse_region(Raw),
            local_stream_buffer:update_region(hann_lsb, RE),
            {200, #{status => <<"ok">>}};
        {error, R} -> bad_request(R)
    end;

dispatch(stream_results, _Req) ->
    Events  = local_stream_buffer:fetch_results(hann_lsb),
    Payload = [serialize_joint_event(E) || E <- Events],
    {200, #{events => Payload, count => length(Payload)}};

dispatch(stream_counters, _Req) ->
    {200, local_stream_buffer:counters(hann_lsb)};

dispatch(_Unknown, _Req) ->
    {404, #{error => <<"not_found">>}}.

%%====================================================================
%% Frame parsers
%%====================================================================

parse_hypntyz(Raw) ->
    #{event_id         => maps:get(<<"event_id">>, Raw, gen_event_id()),
      phenomenon_time  => maps:get(<<"phenomenon_time">>, Raw,
                                   erlang:system_time(microsecond)),
      speed_fft        => to_float_list(maps:get(<<"speed_fft">>,        Raw, [])),
      accel_spectrum   => to_float_list(maps:get(<<"accel_spectrum">>,   Raw, [])),
      motion_harmonics => to_float_list(maps:get(<<"motion_harmonics">>, Raw, [])),
      metadata         => maps:get(<<"metadata">>, Raw, #{})}.

parse_shek3m(Raw) ->
    #{event_id          => maps:get(<<"event_id">>, Raw, gen_event_id()),
      phenomenon_time   => maps:get(<<"phenomenon_time">>, Raw,
                                    erlang:system_time(microsecond)),
      corridor_fft      => to_float_list(maps:get(<<"corridor_fft">>,      Raw, [])),
      h3_adjacency_fft  => to_float_list(maps:get(<<"h3_adjacency_fft">>,  Raw, [])),
      route_coherence   => to_float(maps:get(<<"route_coherence">>,        Raw, 0.0)),
      metadata          => maps:get(<<"metadata">>, Raw, #{})}.

parse_region(Raw) ->
    #{urban_rural_vector  => to_float_list(maps:get(<<"urban_rural_vector">>,  Raw, [])),
      density_embedding   => to_float_list(maps:get(<<"density_embedding">>,   Raw, [])),
      infrastructure_bias => to_float(maps:get(<<"infrastructure_bias">>,      Raw, 0.5)),
      updated_at          => maps:get(<<"updated_at">>, Raw,
                                       erlang:system_time(microsecond))}.

%%====================================================================
%% Serialisers
%%====================================================================

serialize_joint_event(E) ->
    #{event_id             => maps:get(event_id,             E, null),
      phenomenon_time      => maps:get(phenomenon_time,      E, null),
      alignment_confidence => maps:get(alignment_confidence, E, null),
      temporal_fft         => maps:get(temporal_fft,         E, #{}),
      geometric_fft        => maps:get(geometric_fft,        E, #{}),
      region_embedding     => maps:get(region_embedding,     E, null)}.

serialize_neighbor(N) ->
    #{node_id    => maps:get(node_id,    N, null),
      distance   => maps:get(distance,   N, null),
      similarity => maps:get(similarity, N, null),
      rank       => maps:get(rank,       N, null),
      h3_index   => maps:get(h3_index,   N, null),
      stop_name  => maps:get(stop_name,  N, null),
      route_ids  => maps:get(route_ids,  N, []),
      region     => maps:get(region,     N, null)}.

%%====================================================================
%% Internal helpers
%%====================================================================

read_json(Req) ->
    {ok, Body, _} = cowboy_req:read_body(Req),
    try {ok, jiffy:decode(Body, [return_maps])}
    catch _:_ -> {error, <<"invalid JSON">>}
    end.

bad_request(Reason) ->
    {400, #{error => ensure_binary(Reason)}}.

to_float_list(List) when is_list(List) -> [to_float(X) || X <- List];
to_float_list(_)                       -> [].

to_float(X) when is_float(X)   -> X;
to_float(X) when is_integer(X) -> float(X);
to_float(_)                    -> 0.0.

atomise_meta(Map) when is_map(Map) ->
    maps:fold(fun(K, V, Acc) ->
        maps:put(safe_atom(K), V, Acc)
    end, #{}, Map);
atomise_meta(_) -> #{}.

safe_atom(B) ->
    try binary_to_existing_atom(B, utf8)
    catch _:_ -> B
    end.

gen_event_id() ->
    <<I:64>> = crypto:strong_rand_bytes(8),
    list_to_binary(io_lib:format("evt-~16.16.0b", [I])).

ensure_binary(B) when is_binary(B) -> B;
ensure_binary(A) when is_atom(A)   -> atom_to_binary(A, utf8);
ensure_binary(T)                   -> list_to_binary(io_lib:format("~p", [T])).
