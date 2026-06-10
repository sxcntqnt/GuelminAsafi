%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — OTP Application Entry Point
%%%
%%% Boots the supervision tree and the Cowboy HTTP listener.
%%% The HNSW index and GRG gen_servers are started under the supervisor.
%%%-------------------------------------------------------------------
-module(himap_hann_app).
-behaviour(application).

-export([start/2, stop/1]).

%%%-------------------------------------------------------------------
%%% Application callbacks
%%%-------------------------------------------------------------------

start(_StartType, _StartArgs) ->
    Port      = application:get_env(himap_hann, port,     8080),
    Dim       = application:get_env(himap_hann, hnsw_dim, 128),
    M         = application:get_env(himap_hann, hnsw_m,   16),
    Ef        = application:get_env(himap_hann, hnsw_ef,  64),
    DistFn    = application:get_env(himap_hann, hnsw_dist_fn, cosine),

    %% Cowboy HTTP dispatch table
    Dispatch = cowboy_router:compile([
        {'_', [
            {"/health",           hann_http_handler, #{action => health}},
            {"/v1/node",          hann_http_handler, #{action => add_node}},
            {"/v1/search",        hann_http_handler, #{action => search}},
            {"/v1/grg/query",     hann_http_handler, #{action => grg_query}},
            {"/v1/index/bulk",    hann_http_handler, #{action => bulk_add}}
        ]}
    ]),

    {ok, _} = cowboy:start_clear(
        himap_hann_listener,
        [{port, Port}],
        #{env => #{dispatch => Dispatch}}
    ),

    logger:info("HiMap HANN listening on port ~w  dim=~w  M=~w  ef=~w  dist=~w",
                [Port, Dim, M, Ef, DistFn]),

    HNSWOpts = #{dim => Dim, m => M, ef => Ef, dist_fn => DistFn},
    himap_hann_sup:start_link(HNSWOpts).

stop(_State) ->
    cowboy:stop_listener(himap_hann_listener),
    ok.
