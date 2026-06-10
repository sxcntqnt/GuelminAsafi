%% local_stream_buffer.erl
%%
%% Dual-stream spectral synchronizer with sliding memory windows.
%%
%% Three internal stores:
%%
%%   tab_hypntyz  : ordered_set {timestamp_us, ref} → HypntyzFrame
%%   tab_shek3m   : ordered_set {timestamp_us, ref} → Shek3mFrame
%%   tab_out      : ordered_set {timestamp_us, ref} → JointSpectralEvent
%%
%% Region embedding is NOT a sliding buffer.
%% It is slow state held in the gen_server record, updated explicitly.
%% It acts as a cohesion bias on alignment confidence — never FFT'd.
%%
%% HypntyzFrame:
%%   #{ event_id        => binary(),
%%      phenomenon_time => integer(),     %% unix microseconds
%%      speed_fft       => [float()],     %% FFT bins
%%      accel_spectrum  => [float()],
%%      motion_harmonics => [float()],
%%      metadata        => map() }
%%
%% Shek3mFrame:
%%   #{ event_id         => binary(),
%%      phenomenon_time  => integer(),
%%      corridor_fft     => [float()],
%%      h3_adjacency_fft => [float()],
%%      route_coherence  => float(),
%%      metadata         => map() }
%%
%% RegionEmbedding:
%%   #{ urban_rural_vector  => [float()],
%%      density_embedding   => [float()],
%%      infrastructure_bias => float(),
%%      updated_at          => integer() }
%%
%% JointSpectralEvent:
%%   #{ event_id             => binary(),
%%      phenomenon_time      => integer(),
%%      temporal_fft         => HypntyzFrame,
%%      geometric_fft        => Shek3mFrame,
%%      region_embedding     => RegionEmbedding,
%%      alignment_confidence => float() }      %% 0.0–1.0

-module(local_stream_buffer).
-behaviour(gen_server).

-export([
    start_link/1,
    ingest_hypntyz/2,
    ingest_shek3m/2,
    update_region/2,
    fetch_results/1,
    counters/1,
    stop/1
]).

-export([
    init/1,
    handle_call/3,
    handle_cast/2,
    handle_info/2,
    terminate/2
]).

%%=============================================================================
%% Types
%%=============================================================================

-type timestamp_us() :: integer().
-type ets_key()      :: {timestamp_us(), reference()}.

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
    region_embedding     := region_embedding(),
    alignment_confidence := float()
}.

-type emit_fun()   :: fun((joint_spectral_event()) -> ok).
-type commit_fun() :: fun((ets_key(), term()) -> ok).

-type opts() :: #{
    %% Maximum time difference (us) between Hypntyz and Shek3m frames to be
    %% alignment candidates. Default: 500_000 (500ms)
    delta_time              => timestamp_us(),

    %% Minimum alignment_confidence to emit a JointSpectralEvent. Default: 0.5
    confidence_threshold    => float(),

    %% Maximum number of frames to retain per window. Default: 500
    window_size             => pos_integer(),

    %% Called for every emitted JointSpectralEvent (e.g., forward to GuelminAsafi)
    emit_fun                => emit_fun() | undefined,

    %% Called when a frame is stripped from a window (at-least-once commit hook)
    commit_fun              => commit_fun() | undefined,

    %% Buffer JointSpectralEvents internally for fetch_results/1. Default: true
    buffer_results          => boolean(),

    verbose                 => boolean()
}.

-record(state, {
    tab_hypntyz          :: ets:tid(),
    tab_shek3m           :: ets:tid(),
    tab_out              :: ets:tid(),

    %% Region embedding: slow state, not a sliding buffer
    region_embedding     :: region_embedding() | undefined,

    %% Tuning
    delta_time           :: timestamp_us(),
    confidence_threshold :: float(),
    window_size          :: pos_integer(),

    %% Callbacks
    emit_fun             :: emit_fun() | undefined,
    commit_fun           :: commit_fun() | undefined,
    buffer_results       :: boolean(),

    %% Counters
    count_hypntyz        :: non_neg_integer(),
    count_shek3m         :: non_neg_integer(),
    count_emitted        :: non_neg_integer(),

    verbose              :: boolean()
}).

%%=============================================================================
%% Public API
%%=============================================================================

-spec start_link(opts()) -> {ok, pid()} | {error, term()}.
start_link(Opts) ->
    gen_server:start_link(?MODULE, Opts, []).

%% Ingest a frame from the Hypntyz temporal FFT broadcast stream
-spec ingest_hypntyz(pid(), hypntyz_frame()) -> ok.
ingest_hypntyz(Pid, Frame) ->
    gen_server:cast(Pid, {ingest, hypntyz, Frame}).

%% Ingest a frame from the Shek3m geometric FFT broadcast stream
-spec ingest_shek3m(pid(), shek3m_frame()) -> ok.
ingest_shek3m(Pid, Frame) ->
    gen_server:cast(Pid, {ingest, shek3m, Frame}).

%% Update the region embedding (slow state — not per-event)
-spec update_region(pid(), region_embedding()) -> ok.
update_region(Pid, RegionEmbedding) ->
    gen_server:cast(Pid, {update_region, RegionEmbedding}).

%% Fetch and drain the output buffer
-spec fetch_results(pid()) -> [joint_spectral_event()].
fetch_results(Pid) ->
    gen_server:call(Pid, fetch_results).

-spec counters(pid()) -> map().
counters(Pid) ->
    gen_server:call(Pid, counters).

-spec stop(pid()) -> ok.
stop(Pid) ->
    gen_server:stop(Pid).

%%=============================================================================
%% gen_server callbacks
%%=============================================================================

init(Opts) ->
    TabH = ets:new(lsb_hypntyz, [ordered_set, private]),
    TabS = ets:new(lsb_shek3m,  [ordered_set, private]),
    TabO = ets:new(lsb_out,     [ordered_set, private]),
    State = #state{
        tab_hypntyz          = TabH,
        tab_shek3m           = TabS,
        tab_out              = TabO,
        region_embedding     = undefined,
        delta_time           = maps:get(delta_time, Opts, 500_000),
        confidence_threshold = maps:get(confidence_threshold, Opts, 0.5),
        window_size          = maps:get(window_size, Opts, 500),
        emit_fun             = maps:get(emit_fun, Opts, undefined),
        commit_fun           = maps:get(commit_fun, Opts, undefined),
        buffer_results       = maps:get(buffer_results, Opts, true),
        count_hypntyz        = 0,
        count_shek3m         = 0,
        count_emitted        = 0,
        verbose              = maps:get(verbose, Opts, false)
    },
    {ok, State}.

handle_cast({ingest, hypntyz, Frame}, State) ->
    TS  = maps:get(phenomenon_time, Frame),
    Key = {TS, make_ref()},
    ets:insert(State#state.tab_hypntyz, {Key, Frame}),
    log(State, "ingest hypntyz ts=~p", [TS]),
    State1 = State#state{count_hypntyz = State#state.count_hypntyz + 1},
    %% Attempt alignment: newest Hypntyz frame seeks Shek3m partners
    State2 = run_alignment(hypntyz, Key, State1),
    State3 = enforce_window(State#state.tab_hypntyz, State#state.window_size,
                            hypntyz, State2),
    {noreply, State3};

handle_cast({ingest, shek3m, Frame}, State) ->
    TS  = maps:get(phenomenon_time, Frame),
    Key = {TS, make_ref()},
    ets:insert(State#state.tab_shek3m, {Key, Frame}),
    log(State, "ingest shek3m ts=~p", [TS]),
    State1 = State#state{count_shek3m = State#state.count_shek3m + 1},
    %% Attempt alignment: newest Shek3m frame seeks Hypntyz partners
    State2 = run_alignment(shek3m, Key, State1),
    State3 = enforce_window(State#state.tab_shek3m, State#state.window_size,
                            shek3m, State3),
    {noreply, State3};

handle_cast({update_region, RegionEmbedding}, State) ->
    %% Region is slow state — just replace, no FFT, no alignment trigger
    log(State, "region embedding updated at=~p",
        [maps:get(updated_at, RegionEmbedding, 0)]),
    {noreply, State#state{region_embedding = RegionEmbedding}};

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_call(fetch_results, _From, State) ->
    Events = [E || {_K, E} <- ets:tab2list(State#state.tab_out)],
    %% Sort by phenomenon_time ascending before returning
    Sorted = lists:sort(fun(A, B) ->
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

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, State) ->
    ets:delete(State#state.tab_hypntyz),
    ets:delete(State#state.tab_shek3m),
    ets:delete(State#state.tab_out),
    ok.

%%=============================================================================
%% Alignment Engine
%%=============================================================================
%%
%% When a new frame arrives on either stream, the alignment engine searches
%% the OTHER stream's sliding window for candidates within delta_time.
%%
%% For each candidate pair (HypntyzFrame, Shek3mFrame):
%%   1. Check event_id match (strong signal — same originating event)
%%   2. Compute temporal proximity score
%%   3. Apply region embedding as cohesion bias
%%   4. Compute alignment_confidence
%%   5. If confidence >= threshold → emit JointSpectralEvent
%%
%% This replaces the generic JR1/JR2/JS2 join cases with semantically
%% meaningful alignment logic.

-spec run_alignment(hypntyz | shek3m, ets_key(), #state{}) -> #state{}.
run_alignment(PivotalStream, NewKey, State) ->
    {PivotalTab, ExteriorTab} = case PivotalStream of
        hypntyz -> {State#state.tab_hypntyz, State#state.tab_shek3m};
        shek3m  -> {State#state.tab_shek3m,  State#state.tab_hypntyz}
    end,
    case ets:lookup(PivotalTab, NewKey) of
        [] -> State;
        [{NewKey, NewFrame}] ->
            NewTS = maps:get(phenomenon_time, NewFrame),
            %% Find all exterior frames within delta_time of NewTS
            Candidates = find_candidates(ExteriorTab, NewTS, State#state.delta_time),
            %% For each candidate, compute alignment and maybe emit
            lists:foldl(
                fun({CandKey, CandFrame}, St) ->
                    align_and_emit(PivotalStream, NewFrame, CandFrame,
                                   NewKey, CandKey, St)
                end,
                State,
                Candidates
            )
    end.

%% Find all frames in Tab with |ts - pivot_ts| <= delta_time
-spec find_candidates(ets:tid(), timestamp_us(), timestamp_us()) ->
    [{ets_key(), map()}].
find_candidates(Tab, PivotTS, DeltaTime) ->
    Lo = PivotTS - DeltaTime,
    Hi = PivotTS + DeltaTime,
    %% Walk from first key >= Lo
    StartKey = first_at_or_after(Tab, Lo),
    collect_range(Tab, StartKey, Hi, []).

collect_range(_Tab, '$end_of_table', _Hi, Acc) ->
    lists:reverse(Acc);
collect_range(Tab, Key, Hi, Acc) ->
    {TS, _} = Key,
    case TS > Hi of
        true  -> lists:reverse(Acc);
        false ->
            [{Key, Frame}] = ets:lookup(Tab, Key),
            collect_range(Tab, ets:next(Tab, Key), Hi, [{Key, Frame} | Acc])
    end.

%%=============================================================================
%% Alignment confidence computation
%%=============================================================================
%%
%% confidence = weighted combination of:
%%   - event_id_match    : 1.0 if same event_id, else 0.0   weight: 0.50
%%   - temporal_proximity: 1 - (|dt| / delta_time)           weight: 0.30
%%   - region_bias       : infrastructure_bias from region   weight: 0.20
%%
%% Region embedding acts as a cohesion field:
%%   high infrastructure_bias → lower confidence threshold needed
%%   (dense urban areas have higher signal correlation between FFT streams)

-spec align_and_emit(
    hypntyz | shek3m,
    hypntyz_frame() | shek3m_frame(),
    shek3m_frame()  | hypntyz_frame(),
    ets_key(), ets_key(),
    #state{}
) -> #state{}.
align_and_emit(PivotalStream, PivotalFrame, ExteriorFrame,
               _PivotalKey, _ExteriorKey, State) ->
    {HFrame, SFrame} = case PivotalStream of
        hypntyz -> {PivotalFrame, ExteriorFrame};
        shek3m  -> {ExteriorFrame, PivotalFrame}
    end,

    Confidence = compute_confidence(HFrame, SFrame, State),

    log(State, "alignment confidence=~.3f threshold=~.3f",
        [Confidence, State#state.confidence_threshold]),

    case Confidence >= State#state.confidence_threshold of
        false ->
            State;
        true ->
            %% Build the JointSpectralEvent
            PhenomenonTime = (maps:get(phenomenon_time, HFrame) +
                              maps:get(phenomenon_time, SFrame)) div 2,
            EventId = resolve_event_id(HFrame, SFrame),
            Event = #{
                event_id             => EventId,
                phenomenon_time      => PhenomenonTime,
                temporal_fft         => HFrame,
                geometric_fft        => SFrame,
                region_embedding     => State#state.region_embedding,
                alignment_confidence => Confidence
            },
            emit_event(Event, State)
    end.

-spec compute_confidence(hypntyz_frame(), shek3m_frame(), #state{}) -> float().
compute_confidence(HFrame, SFrame, State) ->
    %% Component 1: event_id match (weight 0.50)
    H_EID = maps:get(event_id, HFrame, undefined),
    S_EID = maps:get(event_id, SFrame, undefined),
    EventIdScore = case H_EID =:= S_EID andalso H_EID =/= undefined of
        true  -> 1.0;
        false -> 0.0
    end,

    %% Component 2: temporal proximity (weight 0.30)
    H_TS = maps:get(phenomenon_time, HFrame),
    S_TS = maps:get(phenomenon_time, SFrame),
    Dt   = abs(H_TS - S_TS),
    TemporalScore = case State#state.delta_time of
        0  -> 1.0;
        DT -> max(0.0, 1.0 - (Dt / DT))
    end,

    %% Component 3: region cohesion bias (weight 0.20)
    %% infrastructure_bias is 0.0–1.0; dense urban = closer to 1.0
    RegionBias = case State#state.region_embedding of
        undefined -> 0.5;   %% neutral when no region state loaded
        R         -> maps:get(infrastructure_bias, R, 0.5)
    end,

    Confidence = (0.50 * EventIdScore) +
                 (0.30 * TemporalScore) +
                 (0.20 * RegionBias),
    min(1.0, max(0.0, Confidence)).

%% Prefer matching event_id; fall back to Hypntyz event_id
-spec resolve_event_id(hypntyz_frame(), shek3m_frame()) -> binary().
resolve_event_id(HFrame, SFrame) ->
    H_EID = maps:get(event_id, HFrame, undefined),
    S_EID = maps:get(event_id, SFrame, undefined),
    case H_EID =:= S_EID of
        true  -> H_EID;
        false -> H_EID   %% Hypntyz is the temporal authority
    end.

%%=============================================================================
%% Emit
%%=============================================================================

-spec emit_event(joint_spectral_event(), #state{}) -> #state{}.
emit_event(Event, State) ->
    TS  = maps:get(phenomenon_time, Event),
    Key = {TS, make_ref()},
    %% Buffer internally if configured
    case State#state.buffer_results of
        true  -> ets:insert(State#state.tab_out, {Key, Event});
        false -> ok
    end,
    %% Forward to GuelminAsafi via emit_fun callback
    case State#state.emit_fun of
        undefined -> ok;
        Fun       -> Fun(Event)
    end,
    log(State, "emit JointSpectralEvent event_id=~s confidence=~.3f",
        [maps:get(event_id, Event), maps:get(alignment_confidence, Event)]),
    State#state{count_emitted = State#state.count_emitted + 1}.

%%=============================================================================
%% Window enforcement
%%=============================================================================
%%
%% Each sliding window is bounded by window_size.
%% When the window exceeds the limit, oldest frames are stripped from the head.
%% The commit_fun is called for each stripped frame (at-least-once hook).

-spec enforce_window(ets:tid(), pos_integer(), hypntyz | shek3m, #state{}) -> #state{}.
enforce_window(Tab, MaxSize, Side, State) ->
    Over = ets:info(Tab, size) - MaxSize,
    case Over > 0 of
        false -> State;
        true  -> drop_oldest(Tab, Over, Side, State)
    end.

drop_oldest(_Tab, 0, _Side, State) -> State;
drop_oldest(Tab, N, Side, State) ->
    case ets:first(Tab) of
        '$end_of_table' ->
            State;
        Key ->
            [{Key, Frame}] = ets:lookup(Tab, Key),
            ets:delete(Tab, Key),
            maybe_commit(State#state.commit_fun, Key, Frame),
            log(State, "window strip ~p key=~p", [Side, Key]),
            drop_oldest(Tab, N - 1, Side, State)
    end.

%%=============================================================================
%% ETS traversal helpers
%%=============================================================================

-spec first_at_or_after(ets:tid(), timestamp_us()) -> ets_key() | '$end_of_table'.
first_at_or_after(Tab, TS) ->
    walk_forward(Tab, ets:first(Tab), TS).

walk_forward(_Tab, '$end_of_table', _TS) ->
    '$end_of_table';
walk_forward(Tab, Key, TS) ->
    {KTS, _} = Key,
    case KTS >= TS of
        true  -> Key;
        false -> walk_forward(Tab, ets:next(Tab, Key), TS)
    end.

%%=============================================================================
%% Internal helpers
%%=============================================================================

maybe_commit(undefined, _Key, _Frame) -> ok;
maybe_commit(Fun, Key, Frame)         -> Fun(Key, Frame).

log(#state{verbose = false}, _Fmt, _Args) -> ok;
log(#state{verbose = true},   Fmt,  Args) ->
    io:format("[LSB] " ++ Fmt ++ "~n", Args).
