%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — Top-level Supervisor
%%%
%%% Strategy: rest_for_one
%%%
%%%   hann_ets            owns ETS tables — crash restarts everything
%%%   hann_scylla         DB load — crash restarts scylla+hnsw+grg+lsb
%%%   hann_hnsw           write gen_server — crash restarts hnsw+grg+lsb
%%%   hann_grg            GRG enrichment — crash restarts grg+lsb
%%%   local_stream_buffer spectral sync — crash restarts only lsb
%%%                       (in-memory windows lost, acceptable;
%%%                        ETS and DB intact, new frames rebuild windows)
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

    LSBOpts = #{
        delta_time           => 500_000,   %% 500 ms
        confidence_threshold => 0.60,
        window_size          => 1000,
        emit_fun             => fun hann_sink:handle_joint_event/1,
        buffer_results       => true,
        verbose              => false
    },

    Children = [
        child(hann_ets,            {hann_ets,            start_link, []}),
        child(hann_scylla,         {hann_scylla,         start_link, []}),
        child(hann_hnsw,           {hann_hnsw,           start_link, [Opts]}),
        child(hann_grg,            {hann_grg,            start_link, [#{}]}),
        child(local_stream_buffer, {local_stream_buffer, start_link,
                                    [hann_lsb, LSBOpts]})
    ],
    {ok, {SupFlags, Children}}.

child(Id, {M, _F, _A} = MFA) ->
    #{id       => Id,
      start    => MFA,
      restart  => permanent,
      shutdown => 5000,
      type     => worker,
      modules  => [M]}.
