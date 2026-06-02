%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — Integration Tests
%%%
%%% Run with:  rebar3 eunit
%%%
%%% ── What changed from write 1 ───────────────────────────────────────
%%%
%%%   • setup/0 now starts hann_ets first (owns the ETS tables that
%%%     hann_hnsw:init/1 reads on startup).
%%%
%%%   • A mock_scylla process registers itself as `hann_scylla` so
%%%     all gen_server:call / cast invocations from hann_hnsw are
%%%     absorbed without a live ScyllaDB instance.
%%%
%%%   • _assertAll macro replaced with proper list-of-descriptors
%%%     pattern — returning [?_assert(…), ?_assert(…)] from a test
%%%     function is valid EUnit and composes correctly with foreach.
%%%
%%%   • Added grg_test_() suite covering meta enrichment.
%%%
%%%   • Added test_concurrent_search/1 exercising the ETS
%%%     read_concurrency path across 10 parallel processes.
%%%-------------------------------------------------------------------
-module(hann_hnsw_tests).
-include_lib("eunit/include/eunit.hrl").

-define(DIM, 16).   %% small dimension for fast tests

%%%-------------------------------------------------------------------
%%% Setup / teardown
%%%-------------------------------------------------------------------

setup() ->
    %% Order matters — mirrors the rest_for_one supervisor chain:
    %%   hann_ets first (creates tables), then hann_hnsw (reads them).
    {ok, EtsPid}  = hann_ets:start_link(),
    ScyllaPid     = start_mock_scylla(),
    Opts = #{dim => ?DIM, m => 8, ef => 32, dist_fn => cosine},
    {ok, HnswPid} = hann_hnsw:start_link(Opts),
    {EtsPid, ScyllaPid, HnswPid}.

teardown({EtsPid, ScyllaPid, HnswPid}) ->
    gen_server:stop(HnswPid),
    stop_mock_scylla(ScyllaPid),
    gen_server:stop(EtsPid).

%%%-------------------------------------------------------------------
%%% HNSW index test suite
%%%-------------------------------------------------------------------

hnsw_test_() ->
    {foreach, fun setup/0, fun teardown/1, [
        fun test_empty_search/1,
        fun test_single_node/1,
        fun test_identity_search/1,
        fun test_k_gt_n/1,
        fun test_cosine_ordering/1,
        fun test_bulk_add/1,
        fun test_duplicate_rejection/1,
        fun test_concurrent_search/1
    ]}.

%%%-------------------------------------------------------------------
%%% Individual HNSW tests
%%%-------------------------------------------------------------------

test_empty_search(_) ->
    Results = hann_hnsw:search(hann_vector:random_unit(?DIM), 5),
    ?_assertEqual([], Results).

test_single_node(_) ->
    V = hann_vector:random_unit(?DIM),
    ok = hann_hnsw:add(1001, V),
    Results = hann_hnsw:search(V, 1),
    ?_assertMatch([{_, 1001}], Results).

test_identity_search(_) ->
    V = hann_vector:random_unit(?DIM),
    ok = hann_hnsw:add(2001, V),
    %% 19 random distractors
    [hann_hnsw:add(2000 + I, hann_vector:random_unit(?DIM))
     || I <- lists:seq(2, 20)],
    [{Dist, NID}] = hann_hnsw:search(V, 1),
    %% Return a list of descriptors — valid EUnit, composes with foreach
    [?_assertEqual(2001, NID),
     ?_assert(Dist < 1.0e-6)].

test_k_gt_n(_) ->
    V = hann_vector:random_unit(?DIM),
    ok = hann_hnsw:add(3001, V),
    ok = hann_hnsw:add(3002, hann_vector:random_unit(?DIM)),
    Results = hann_hnsw:search(V, 100),
    %% Only 2 nodes exist; must not crash or pad with garbage
    ?_assert(length(Results) =< 2).

test_cosine_ordering(_) ->
    Base  = hann_vector:random_unit(?DIM),
    Close = hann_vector:normalize([X + rand:uniform() * 0.01 || X <- Base]),
    Far   = hann_vector:normalize([-X || X <- Base]),   %% anti-parallel → max distance
    ok = hann_hnsw:add(4001, Close),
    ok = hann_hnsw:add(4002, Far),
    [{D1, _}, {D2, _}] = hann_hnsw:search(Base, 2),
    ?_assert(D1 =< D2).

test_bulk_add(_) ->
    Pairs = [{5000 + I, hann_vector:random_unit(?DIM)} || I <- lists:seq(1, 50)],
    {ok, Count} = hann_hnsw:bulk_add(Pairs),
    Stats = hann_hnsw:stats(),
    [?_assertEqual(50, Count),
     ?_assertEqual(50, maps:get(node_count, Stats))].

test_duplicate_rejection(_) ->
    V  = hann_vector:random_unit(?DIM),
    ok = hann_hnsw:add(6001, V),
    {error, already_exists} = hann_hnsw:add(6001, V),
    Stats = hann_hnsw:stats(),
    ?_assertEqual(1, maps:get(node_count, Stats)).

%% @doc Exercises the ETS read_concurrency path.
%%      10 processes call search/2 simultaneously; none should crash
%%      or block each other.  This is the core guarantee of write 2.
test_concurrent_search(_) ->
    ok = hann_hnsw:add(8001, hann_vector:random_unit(?DIM)),
    ok = hann_hnsw:add(8002, hann_vector:random_unit(?DIM)),
    Query = hann_vector:random_unit(?DIM),
    Monitors = [spawn_monitor(fun() ->
                    R = hann_hnsw:search(Query, 2),
                    exit({ok, R})
                end) || _ <- lists:seq(1, 10)],
    Outcomes = [receive
                    {'DOWN', Ref, process, Pid, {ok, R}} -> R
                after 2000 ->
                    timeout
                end || {Pid, Ref} <- Monitors],
    ?_assert(lists:all(fun(R) -> is_list(R) end, Outcomes)).

%%%-------------------------------------------------------------------
%%% GRG enrichment test suite
%%%-------------------------------------------------------------------

grg_test_() ->
    Setup = fun() ->
        {ok, EtsPid}  = hann_ets:start_link(),
        ScyllaPid     = start_mock_scylla(),
        Opts = #{dim => ?DIM, m => 8, ef => 32, dist_fn => cosine},
        {ok, HnswPid} = hann_hnsw:start_link(Opts),
        {ok, GrgPid}  = hann_grg:start_link(#{}),
        {EtsPid, ScyllaPid, HnswPid, GrgPid}
    end,
    Teardown = fun({EtsPid, ScyllaPid, HnswPid, GrgPid}) ->
        gen_server:stop(GrgPid),
        gen_server:stop(HnswPid),
        stop_mock_scylla(ScyllaPid),
        gen_server:stop(EtsPid)
    end,
    {foreach, Setup, Teardown, [
        fun test_grg_empty/1,
        fun test_grg_meta_enrichment/1
    ]}.

test_grg_empty(_) ->
    Results = hann_grg:query_neighbors(hann_vector:random_unit(?DIM), 3),
    ?_assertEqual([], Results).

test_grg_meta_enrichment(_) ->
    V  = hann_vector:random_unit(?DIM),
    ok = hann_hnsw:add(7001, V),
    ok = hann_grg:register_node_meta(7001, #{
             h3_index  => <<"8a182da6a4c7fff">>,
             stop_name => <<"GPO Nairobi">>,
             route_ids => [42, 107],
             region    => <<"Nairobi CBD">>
         }),
    [N | _] = hann_grg:query_neighbors(V, 1),
    [?_assertEqual(7001,                  maps:get(node_id,   N)),
     ?_assertEqual(1,                     maps:get(rank,      N)),
     ?_assertEqual(<<"GPO Nairobi">>,     maps:get(stop_name, N)),
     ?_assertEqual(<<"8a182da6a4c7fff">>, maps:get(h3_index,  N)),
     ?_assertEqual([42, 107],             maps:get(route_ids, N))].

%%%-------------------------------------------------------------------
%%% Vector math unit tests  (no ETS or ScyllaDB needed)
%%%-------------------------------------------------------------------

vector_test_() ->
    [
        ?_assert(abs(hann_vector:norm(hann_vector:normalize([3.0, 4.0])) - 1.0) < 1.0e-9),
        ?_assert(abs(hann_vector:dot([1.0, 0.0], [0.0, 1.0]))                   < 1.0e-9),
        ?_assert(abs(hann_vector:cosine([1.0, 0.0], [0.0, 1.0]) - 1.0)         < 1.0e-6),
        ?_assert(abs(hann_vector:cosine([1.0, 0.0], [1.0, 0.0]))                < 1.0e-9),
        ?_assert(abs(hann_vector:euclidean([0.0, 0.0], [3.0, 4.0]) - 5.0)      < 1.0e-9)
    ].

%%%-------------------------------------------------------------------
%%% Mock ScyllaDB
%%%
%%% Registers as `hann_scylla` and absorbs all gen_server:call and
%%% gen_server:cast messages so tests run without a live ScyllaDB.
%%%-------------------------------------------------------------------

start_mock_scylla() ->
    Pid = spawn(fun mock_scylla_loop/0),
    register(hann_scylla, Pid),
    Pid.

stop_mock_scylla(Pid) ->
    unregister(hann_scylla),
    exit(Pid, kill).

mock_scylla_loop() ->
    receive
        %% gen_server:call — reply ok to caller
        {'$gen_call', {Caller, Tag}, _Req} ->
            Caller ! {Tag, ok};
        %% gen_server:cast and anything else — swallow silently
        _ ->
            ok
    end,
    mock_scylla_loop().
