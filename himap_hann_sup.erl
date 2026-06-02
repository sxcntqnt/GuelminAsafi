%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — Top-level Supervisor
%%%
%%% Strategy: rest_for_one — each child depends on those above it.
%%%
%%%   hann_ets    — owns the 3 named ETS tables.
%%%                 If it crashes, ALL children restart (tables gone).
%%%
%%%   hann_scylla — marina connection pool + startup DB load.
%%%                 If it crashes, hann_scylla + hann_hnsw + hann_grg
%%%                 restart; scylla reloads ETS from ScyllaDB first.
%%%
%%%   hann_hnsw   — write gen_server (serialised inserts via ETS).
%%%                 If it crashes, only hann_hnsw + hann_grg restart;
%%%                 ETS is still valid so no DB reload is needed.
%%%
%%%   hann_grg    — GRG enrichment layer; depends on hann_hnsw search.
%%%-------------------------------------------------------------------
-module(himap_hann_sup).
-behaviour(supervisor).

-export([start_link/1, init/1]).

start_link(Opts) ->
    supervisor:start_link({local, ?MODULE}, ?MODULE, Opts).

init(Opts) ->
    SupFlags = #{
        strategy  => rest_for_one,
        intensity => 5,
        period    => 10
    },
    Children = [
        child(hann_ets,    {hann_ets,    start_link, []}),
        child(hann_scylla, {hann_scylla, start_link, []}),
        child(hann_hnsw,   {hann_hnsw,   start_link, [Opts]}),
        child(hann_grg,    {hann_grg,    start_link, [#{}]})
    ],
    {ok, {SupFlags, Children}}.

child(Id, {M, _F, _A} = MFA) ->
    #{id       => Id,
      start    => MFA,
      restart  => permanent,
      shutdown => 5000,
      type     => worker,
      modules  => [M]}.
