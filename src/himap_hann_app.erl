%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — OTP Application Entry Point
%%%-------------------------------------------------------------------
-module(himap_hann_app).
-behaviour(application).

-export([start/2, stop/1]).

start(_StartType, _StartArgs) ->
    Port   = application:get_env(himap_hann, port,         9950),
    Dim    = application:get_env(himap_hann, hnsw_dim,     128),
    M      = application:get_env(himap_hann, hnsw_m,       16),
    Ef     = application:get_env(himap_hann, hnsw_ef,      64),
    DistFn = application:get_env(himap_hann, hnsw_dist_fn, cosine),

    Dispatch = cowboy_router:compile([{'_', [
        %% ── Core index ─────────────────────────────────────────────
        {"/health",              hann_http_handler, #{action => health}},
        {"/v1/node",             hann_http_handler, #{action => add_node}},
        {"/v1/index/bulk",       hann_http_handler, #{action => bulk_add}},
        {"/v1/search",           hann_http_handler, #{action => search}},

        %% ── GRG ────────────────────────────────────────────────────
        {"/v1/grg/query",        hann_http_handler, #{action => grg_query}},
        {"/v1/grg/meta",         hann_http_handler, #{action => register_meta}},

        %% ── LocalStreamBuffer — spectral ingestion ─────────────────
        {"/v1/stream/hypntyz",   hann_http_handler, #{action => stream_hypntyz}},
        {"/v1/stream/shek3m",    hann_http_handler, #{action => stream_shek3m}},
        {"/v1/stream/region",    hann_http_handler, #{action => stream_region}},
        {"/v1/stream/results",   hann_http_handler, #{action => stream_results}},
        {"/v1/stream/counters",  hann_http_handler, #{action => stream_counters}}
    ]}]),

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
