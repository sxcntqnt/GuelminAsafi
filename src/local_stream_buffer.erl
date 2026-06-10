%%%-------------------------------------------------------------------
%%% @doc LocalStreamBuffer — dual-stream spectral synchroniser
%%%
%%% ── Storage layout ────────────────────────────────────────────────
%%%
%%%   tab_hypntyz_time   ordered_set  {TS, Seq} → HypntyzFrame
%%%   tab_shek3m_time    ordered_set  {TS, Seq} → Shek3mFrame
%%%   tab_hypntyz_event  set          EventID   → {TS, Seq}
%%%   tab_shek3m_event   set          EventID   → {TS, Seq}
%%%   tab_out            ordered_set  {TS, Seq} → JointSpectralEvent
%%%
%%%   Region embedding is gen_server record state (slow, no sliding).
%%%
%%% ── Alignment complexity ─────────────────────────────────────────
%%%
%%%   Exact path  (event_id match)  O(1)   — event index hash lookup
%%%   Fuzzy path  (no id match)     O(k)   — k = frames in ±delta_time
%%%   Window seek                   O(log n) — ets:next sentinel trick
%%%
%%% ── Key design ────────────────────────────────────────────────────
%%%
%%%   Keys are {Timestamp_us, Seq} where
%%%     Seq = erlang:unique_integer([monotonic])
%%%
%%%   Benefits over {Timestamp, make_ref()}:
%%%     • fully ordered (no atom/ref term-order surprises)
%%%     • no heavyweight reference objects
%%%     • allows O(log n) lower-bound seek via sentinel tuple
%%%
%%% ── Sentinel seek ─────────────────────────────────────────────────
%%%
%%%   To find the first key with TS >= Lo in an ordered_set:
%%%
%%%     ets:next(Tab, {Lo - 1, []})
%%%
%%%   Works because [] (list) > integer in Erlang term order,
%%%   so {Lo-1, []} is strictly greater than any {Lo-1, integer}
%%%   key, making it a valid predecessor sentinel.
%%%   ets:next/2 on ordered_set accepts non-existent keys.
%%%-------------------------------------------------------------------
-module(local_stream_buffer).
-behaviour(gen_server).

-export([
    start_link/1,
    start_link/2,
    ingest_hypntyz/2,
    ingest_shek3m/2,
    update_region/2,
    fetch_results/1,
    counters/1,
    stop/1
]).

-export([init/1, handle_call/3, handle_cast/2,
         handle_info/2, terminate/2]).

%%====================================================================
%% Types
%%====================================================================

-type timestamp_us() :: integer().
-type seq()          :: integer().           %% erlang:unique_integer([monotonic])
-type ets_key()      :: {timestamp_us(), seq()}.

-type hypntyz_frame() :: #{
    event_id         := binary(),
    phenomenon_time  := timestamp_us(),
    speed_fft        := [float()],
    accel_spectrum   := [float()],
    motion_harmonics := [float()],
    metadata         => map()
}.

-type shek3m_frame() :: #{
    event_id          := binary(),
    phenomenon_time   := timestamp_us(),
    corridor_fft      := [float()],
    h3_adjacency_fft  := [float()],
    route_coherence   := float(),
    metadata          => map()
}.

-type region_embedding() :: #{
    urban_rural_vector  := [float()],
    density_embedding   := [float()],
    infrastructure_bias := float(),
    updated_at          := timestamp_us()
}.

-type joint_spectral_event() :: #{
    event_id             := binary(),
    phenomenon_time      := timestamp_us(),
    temporal_fft         := hypntyz_frame(),
    geometric_fft        := shek3m_frame(),
    region_embedding     := region_embedding() | undefined,
    alignment_confidence := float()
}.

-type emit_fun()   :: fun((joint_spectral_event()) -> ok).
-type commit_fun() :: fun((ets_key(), term()) -> ok).

-type opts() :: #{
    delta_time           => timestamp_us(),
    confidence_threshold => float(),
    window_size          => pos_integer(),
    emit_fun             => emit_fun() | undefined,
    commit_fun           => commit_fun() | undefined,
    buffer_results       => boolean(),
    verbose              => boolean()
}.

-export_type([
    hypntyz_frame/0, shek3m_frame/0,
    region_embedding/0, joint_spectral_event/0,
    opts/0
]).

-record(state, {
    %% Time-series indexes (ordered_set, {TS,Seq} → Frame)
    tab_hypntyz_time     :: ets:tid(),
    tab_shek3m_time      :: ets:tid(),

    %% Event-id indexes (set, EventID → {TS,Seq} of latest frame)
    tab_hypntyz_event    :: ets:tid(),
    tab_shek3m_event     :: ets:tid(),

    %% Output buffer
    tab_out              :: ets:tid(),

    %% Region embedding: slow state, not a sliding buffer
    region_embedding     :: region_embedding() | undefined,

    delta_time           :: timestamp_us(),
    confidence_threshold :: float(),
    window_size          :: pos_integer(),

    emit_fun             :: emit_fun() | undefined,
    commit_fun           :: commit_fun() | undefined,
    buffer_results       :: boolean(),

    count_hypntyz        :: non_neg_integer(),
    count_shek3m         :: non_neg_integer(),
    count_emitted        :: non_neg_integer(),

    verbose              :: boolean()
}).

%%====================================================================
%% Public API
%%====================================================================

-spec start_link(opts()) -> {ok, pid()} | {error, term()}.
start_link(Opts) ->
    gen_server:start_link(?MODULE, Opts, []).

%% Named variant for supervised use (e.g. start_link(hann_lsb, Opts))
-spec start_link(atom(), opts()) -> {ok, pid()} | {error, term()}.
start_link(Name, Opts) ->
    gen_server:start_link({local, Name}, ?MODULE, Opts, []).

-spec ingest_hypntyz(pid() | atom(), hypntyz_frame()) -> ok.
ingest_hypntyz(Srv, Frame) ->
    gen_server:cast(Srv, {ingest, hypntyz, Frame}).

-spec ingest_shek3m(pid() | atom(), shek3m_frame()) -> ok.
ingest_shek3m(Srv, Frame) ->
    gen_server:cast(Srv, {ingest, shek3m, Frame}).

-spec update_region(pid() | atom(), region_embedding()) -> ok.
update_region(Srv, RE) ->
    gen_server:cast(Srv, {update_region, RE}).

-spec fetch_results(pid() | atom()) -> [joint_spectral_event()].
fetch_results(Srv) ->
    gen_server:call(Srv, fetch_results).

-spec counters(pid() | atom()) -> map().
counters(Srv) ->
    gen_server:call(Srv, counters).

-spec stop(pid() | atom()) -> ok.
stop(Srv) ->
    gen_server:stop(Srv).

%%====================================================================
%% gen_server callbacks
%%====================================================================

init(Opts) ->
    TabHT = ets:new(lsb_hypntyz_time,  [ordered_set, private]),
    TabST = ets:new(lsb_shek3m_time,   [ordered_set, private]),
    TabHE = ets:new(lsb_hypntyz_event, [set, private]),
    TabSE = ets:new(lsb_shek3m_event,  [set, private]),
    TabO  = ets:new(lsb_out,           [ordered_set, private]),
    {ok, #state{
        tab_hypntyz_time     = TabHT,
        tab_shek3m_time      = TabST,
        tab_hypntyz_event    = TabHE,
        tab_shek3m_event     = TabSE,
        tab_out              = TabO,
        region_embedding     = undefined,
        delta_time           = maps:get(delta_time,           Opts, 500_000),
        confidence_threshold = maps:get(confidence_threshold, Opts, 0.5),
        window_size          = maps:get(window_size,          Opts, 500),
        emit_fun             = maps:get(emit_fun,             Opts, undefined),
        commit_fun           = maps:get(commit_fun,           Opts, undefined),
        buffer_results       = maps:get(buffer_results,       Opts, true),
        count_hypntyz        = 0,
        count_shek3m         = 0,
        count_emitted        = 0,
        verbose              = maps:get(verbose,              Opts, false)
    }}.

handle_cast({ingest, hypntyz, Frame}, State) ->
    TS  = maps:get(phenomenon_time, Frame),
    Key = {TS, erlang:unique_integer([monotonic])},
    ets:insert(State#state.tab_hypntyz_time, {Key, Frame}),
    index_event(State#state.tab_hypntyz_event,
                maps:get(event_id, Frame, undefined), Key),
    log(State, "ingest hypntyz ts=~w", [TS]),
    S1 = State#state{count_hypntyz = State#state.count_hypntyz + 1},
    S2 = run_alignment(hypntyz, Key, Frame, S1),
    S3 = enforce_window(State#state.tab_hypntyz_time,
                        State#state.tab_hypntyz_event,
                        State#state.window_size, hypntyz, S2),
    {noreply, S3};

handle_cast({ingest, shek3m, Frame}, State) ->
    TS  = maps:get(phenomenon_time, Frame),
    Key = {TS, erlang:unique_integer([monotonic])},
    ets:insert(State#state.tab_shek3m_time, {Key, Frame}),
    index_event(State#state.tab_shek3m_event,
                maps:get(event_id, Frame, undefined), Key),
    log(State, "ingest shek3m ts=~w", [TS]),
    S1 = State#state{count_shek3m = State#state.count_shek3m + 1},
    S2 = run_alignment(shek3m, Key, Frame, S1),
    S3 = enforce_window(State#state.tab_shek3m_time,
                        State#state.tab_shek3m_event,
                        State#state.window_size, shek3m, S2),
    {noreply, S3};

handle_cast({update_region, RE}, State) ->
    log(State, "region updated at=~w", [maps:get(updated_at, RE, 0)]),
    {noreply, State#state{region_embedding = RE}};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_call(fetch_results, _From, State) ->
    Events = [E || {_K, E} <- ets:tab2list(State#state.tab_out)],
    Sorted = lists:sort(
        fun(A, B) ->
            maps:get(phenomenon_time, A) =< maps:get(phenomenon_time, B)
        end, Events),
    ets:delete_all_objects(State#state.tab_out),
    {reply, Sorted, State};

handle_call(counters, _From, State) ->
    {reply, #{
        hypntyz => State#state.count_hypntyz,
        shek3m  => State#state.count_shek3m,
        emitted => State#state.count_emitted
    }, State};

handle_call(_Req, _From, State) ->
    {reply, {error, unknown}, State}.

handle_info(_Info, State) -> {noreply, State}.

terminate(_Reason, State) ->
    ets:delete(State#state.tab_hypntyz_time),
    ets:delete(State#state.tab_shek3m_time),
    ets:delete(State#state.tab_hypntyz_event),
    ets:delete(State#state.tab_shek3m_event),
    ets:delete(State#state.tab_out),
    ok.

%%====================================================================
%% Alignment Engine
%%====================================================================
%%
%% Two-phase strategy:
%%
%%  EXACT  O(1) — event index lookup.
%%         Same event_id in both streams → align immediately.
%%         Accounts for 0.50 weight in confidence.
%%
%%  FUZZY  O(k) — temporal range scan, k = frames in ±delta_time.
%%         Only applied to frames whose event_id ≠ pivot event_id
%%         (exact-matched frames are excluded to prevent double-emit).
%%
%%  Early-exit — if the newest frame in the exterior window predates
%%  (pivot_ts - delta_time), skip the fuzzy path entirely.

-spec run_alignment(hypntyz | shek3m, ets_key(), map(), #state{}) -> #state{}.
run_alignment(Stream, Key, Frame, State) ->
    EID = maps:get(event_id, Frame, undefined),
    {ExtTimeTab, ExtEventTab} = exterior_tabs(Stream, State),
    NewTS = maps:get(phenomenon_time, Frame),

    %% ── Phase 1: exact event-id lookup ──────────────────────────────
    {State1, ExactKey} = case exact_match(EID, ExtEventTab, ExtTimeTab) of
        {found, CandKey, CandFrame} ->
            St = align_and_emit(Stream, Frame, CandFrame, Key, CandKey, State),
            {St, CandKey};
        not_found ->
            {State, undefined}
    end,

    %% ── Phase 2: fuzzy temporal scan (early-exit check first) ───────
    case needs_fuzzy_scan(ExtTimeTab, NewTS, State#state.delta_time) of
        false ->
            State1;
        true ->
            Candidates = find_candidates(ExtTimeTab, NewTS, State#state.delta_time),
            %% Exclude: the frame from exact path + any frame with same event_id
            Filtered = [P || {CK, CF} = P <- Candidates,
                             CK =/= ExactKey,
                             maps:get(event_id, CF, undefined) =/= EID],
            lists:foldl(
                fun({CK, CF}, St) ->
                    align_and_emit(Stream, Frame, CF, Key, CK, St)
                end,
                State1,
                Filtered
            )
    end.

%% O(1) event-id lookup into exterior stream's event index
exact_match(undefined, _EventTab, _TimeTab) ->
    not_found;
exact_match(EID, EventTab, TimeTab) ->
    case ets:lookup(EventTab, EID) of
        [{EID, CandKey}] ->
            case ets:lookup(TimeTab, CandKey) of
                [{CandKey, CandFrame}] -> {found, CandKey, CandFrame};
                []                     -> not_found
            end;
        [] ->
            not_found
    end.

%% Early-exit: if the newest exterior frame is older than
%% (pivot_ts - delta_time), no temporal candidates exist.
needs_fuzzy_scan(Tab, PivotTS, DeltaTime) ->
    case ets:last(Tab) of
        '$end_of_table' -> false;
        {LastTS, _}     -> LastTS >= (PivotTS - DeltaTime)
    end.

%%====================================================================
%% Temporal range scan  O(log n + k)
%%====================================================================

%% Find all frames in Tab with |ts - PivotTS| <= DeltaTime.
%% Seek is O(log n) via the sentinel; iteration is O(k).
-spec find_candidates(ets:tid(), timestamp_us(), timestamp_us()) ->
    [{ets_key(), map()}].
find_candidates(Tab, PivotTS, DeltaTime) ->
    Lo = PivotTS - DeltaTime,
    Hi = PivotTS + DeltaTime,
    %% Sentinel: {Lo-1, []} > any {Lo-1, integer} because list > integer
    %% in Erlang term order.  ets:next on ordered_set accepts any term.
    StartKey = ets:next(Tab, {Lo - 1, []}),
    collect_range(Tab, StartKey, Hi, []).

collect_range(_Tab, '$end_of_table', _Hi, Acc) ->
    lists:reverse(Acc);
collect_range(Tab, {TS, _} = Key, Hi, Acc) when TS =< Hi ->
    [{Key, Frame}] = ets:lookup(Tab, Key),
    collect_range(Tab, ets:next(Tab, Key), Hi, [{Key, Frame} | Acc]);
collect_range(_Tab, _Key, _Hi, Acc) ->
    lists:reverse(Acc).

%%====================================================================
%% Confidence computation
%%====================================================================
%%
%%  confidence = 0.50 * event_id_score
%%             + 0.30 * temporal_proximity_score
%%             + 0.20 * region_cohesion_bias
%%
%%  event_id_score    : 1.0 if same id, else 0.0
%%  temporal_proximity: 1 - |dt| / delta_time
%%  region_bias       : infrastructure_bias from region embedding
%%                      (0.5 if no region state loaded)

-spec align_and_emit(
    hypntyz | shek3m, map(), map(),
    ets_key(), ets_key(), #state{}) -> #state{}.
align_and_emit(Stream, PivotFrame, ExteriorFrame,
               _PK, _EK, State) ->
    {HFrame, SFrame} = case Stream of
        hypntyz -> {PivotFrame,   ExteriorFrame};
        shek3m  -> {ExteriorFrame, PivotFrame}
    end,
    Confidence = compute_confidence(HFrame, SFrame, State),
    log(State, "align conf=~.3f threshold=~.3f",
        [Confidence, State#state.confidence_threshold]),
    case Confidence >= State#state.confidence_threshold of
        false -> State;
        true  ->
            PT = (maps:get(phenomenon_time, HFrame) +
                  maps:get(phenomenon_time, SFrame)) div 2,
            Event = #{
                event_id             => resolve_event_id(HFrame, SFrame),
                phenomenon_time      => PT,
                temporal_fft         => HFrame,
                geometric_fft        => SFrame,
                region_embedding     => State#state.region_embedding,
                alignment_confidence => Confidence
            },
            emit_event(Event, State)
    end.

-spec compute_confidence(map(), map(), #state{}) -> float().
compute_confidence(HFrame, SFrame, State) ->
    H_EID = maps:get(event_id, HFrame, undefined),
    S_EID = maps:get(event_id, SFrame, undefined),
    EventScore = case H_EID =:= S_EID andalso H_EID =/= undefined of
        true  -> 1.0;
        false -> 0.0
    end,

    H_TS = maps:get(phenomenon_time, HFrame),
    S_TS = maps:get(phenomenon_time, SFrame),
    Dt   = abs(H_TS - S_TS),
    TemporalScore = case State#state.delta_time of
        0  -> 1.0;
        DT -> max(0.0, 1.0 - (Dt / DT))
    end,

    RegionBias = case State#state.region_embedding of
        undefined -> 0.5;
        R         -> maps:get(infrastructure_bias, R, 0.5)
    end,

    min(1.0, max(0.0,
        (0.50 * EventScore) + (0.30 * TemporalScore) + (0.20 * RegionBias)
    )).

resolve_event_id(HFrame, SFrame) ->
    H_EID = maps:get(event_id, HFrame, undefined),
    S_EID = maps:get(event_id, SFrame, undefined),
    case H_EID =:= S_EID of
        true  -> H_EID;
        false -> H_EID   %% Hypntyz is the temporal authority
    end.

%%====================================================================
%% Emit
%%====================================================================

-spec emit_event(joint_spectral_event(), #state{}) -> #state{}.
emit_event(Event, State) ->
    TS  = maps:get(phenomenon_time, Event),
    Key = {TS, erlang:unique_integer([monotonic])},
    case State#state.buffer_results of
        true  -> ets:insert(State#state.tab_out, {Key, Event});
        false -> ok
    end,
    case State#state.emit_fun of
        undefined -> ok;
        Fun       -> spawn(fun() -> Fun(Event) end)
    end,
    log(State, "emit event_id=~s conf=~.3f",
        [maps:get(event_id, Event, <<"?">>),
         maps:get(alignment_confidence, Event, 0.0)]),
    State#state{count_emitted = State#state.count_emitted + 1}.

%%====================================================================
%% Window enforcement
%%====================================================================

-spec enforce_window(ets:tid(), ets:tid(), pos_integer(),
                     hypntyz | shek3m, #state{}) -> #state{}.
enforce_window(TimeTab, EventTab, MaxSize, Side, State) ->
    Excess = ets:info(TimeTab, size) - MaxSize,
    case Excess > 0 of
        false -> State;
        true  -> drop_oldest(TimeTab, EventTab, Excess, Side, State)
    end.

drop_oldest(_TT, _ET, 0, _Side, State) -> State;
drop_oldest(TimeTab, EventTab, N, Side, State) ->
    case ets:first(TimeTab) of
        '$end_of_table' ->
            State;
        Key ->
            [{Key, Frame}] = ets:lookup(TimeTab, Key),
            ets:delete(TimeTab, Key),
            remove_event_index(EventTab,
                               maps:get(event_id, Frame, undefined),
                               Key),
            maybe_commit(State#state.commit_fun, Key, Frame),
            log(State, "strip ~w key=~w", [Side, Key]),
            drop_oldest(TimeTab, EventTab, N - 1, Side, State)
    end.

%%====================================================================
%% Internal helpers
%%====================================================================

%% Insert into event index only when event_id is defined.
%% Latest frame always overwrites (last-write-wins per event_id).
index_event(_Tab, undefined, _Key) -> ok;
index_event(Tab, EID, Key)         -> ets:insert(Tab, {EID, Key}).

%% Remove event index entry only if it still points to the key being dropped.
remove_event_index(_Tab, undefined, _Key) -> ok;
remove_event_index(Tab, EID, Key) ->
    case ets:lookup(Tab, EID) of
        [{EID, Key}] -> ets:delete(Tab, EID);
        _            -> ok   %% newer frame has already updated the index
    end.

%% Return the exterior stream's {time_tab, event_tab}
exterior_tabs(hypntyz, State) ->
    {State#state.tab_shek3m_time, State#state.tab_shek3m_event};
exterior_tabs(shek3m, State) ->
    {State#state.tab_hypntyz_time, State#state.tab_hypntyz_event}.

maybe_commit(undefined, _Key, _Frame) -> ok;
maybe_commit(Fun, Key, Frame)         -> Fun(Key, Frame).

log(#state{verbose = false}, _Fmt, _Args) -> ok;
log(#state{verbose = true},   Fmt,  Args) ->
    io:format("[LSB] " ++ Fmt ++ "~n", Args).
