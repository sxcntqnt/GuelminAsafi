%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — GuelminAsafi Sink
%%%
%%% Bridges local_stream_buffer → hann_hnsw / hann_grg.
%%%
%%% Every JointSpectralEvent that clears the alignment confidence
%%% threshold arrives here via the emit_fun callback.  This module:
%%%
%%%   1. Assembles a 128-d RegionDCL vector from the spectral fields.
%%%   2. Derives a stable integer node ID from the event.
%%%   3. Calls hann_hnsw:add/2  to index the vector.
%%%   4. Calls hann_grg:register_node_meta/2 to store spatial metadata.
%%%
%%% ── Vector layout (128 dims) ────────────────────────────────────────
%%%
%%%   Slot   0-31  (32) speed_fft           HypntyzFrame
%%%   Slot  32-51  (20) accel_spectrum       HypntyzFrame
%%%   Slot  52-67  (16) motion_harmonics     HypntyzFrame
%%%   Slot  68-91  (24) corridor_fft         Shek3mFrame
%%%   Slot  92-107 (16) h3_adjacency_fft     Shek3mFrame
%%%   Slot  108     (1) route_coherence       Shek3mFrame
%%%   Slot 109-116  (8) urban_rural_vector   RegionEmbedding
%%%   Slot 117-124  (8) density_embedding    RegionEmbedding
%%%   Slot  125     (1) infrastructure_bias  RegionEmbedding
%%%   Slot  126     (1) alignment_confidence JointSpectralEvent
%%%   Slot  127     (1) 0.0                  reserved
%%%
%%% ── Node ID derivation ──────────────────────────────────────────────
%%%
%%%   erlang:phash2({EventID, PhenomenonTime bsr 20}, 16#0FFFFFFF)
%%%
%%%   bsr 20 on microsecond timestamps gives ~1-second buckets.
%%%   Frames from the same event arriving within one second map to
%%%   the same node — updates metadata without inserting duplicates.
%%%-------------------------------------------------------------------
-module(hann_sink).

-export([handle_joint_event/1, event_to_vector/1, event_to_node_id/1]).

%%====================================================================
%% Public
%%====================================================================

%% @doc emit_fun callback — called by local_stream_buffer for every
%%      JointSpectralEvent that clears the confidence threshold.
%%      Runs in a spawned process (see emit_event/2 in LSB).
-spec handle_joint_event(map()) -> ok.
handle_joint_event(Event) ->
    NodeID = event_to_node_id(Event),
    Vector = event_to_vector(Event),
    Meta   = event_to_meta(Event),

    case hann_hnsw:add(NodeID, Vector) of
        ok ->
            hann_grg:register_node_meta(NodeID, Meta),
            logger:debug("[hann_sink] indexed node=~w conf=~.3f",
                         [NodeID, maps:get(alignment_confidence, Event, 0.0)]);
        {error, already_exists} ->
            %% Same event, same ~1s bucket — just refresh metadata
            hann_grg:register_node_meta(NodeID, Meta)
    end,
    ok.

%% @doc Build the 128-d RegionDCL vector from a JointSpectralEvent.
-spec event_to_vector(map()) -> [float()].
event_to_vector(Event) ->
    HF = maps:get(temporal_fft,     Event, #{}),
    SF = maps:get(geometric_fft,    Event, #{}),
    RE = maps:get(region_embedding, Event, #{}),
    AC = float_val(maps:get(alignment_confidence, Event, 0.0)),

    %% {SourceList, BudgetDims}
    Segments = [
        {flist(maps:get(speed_fft,           HF, [])), 32},
        {flist(maps:get(accel_spectrum,       HF, [])), 20},
        {flist(maps:get(motion_harmonics,     HF, [])), 16},
        {flist(maps:get(corridor_fft,         SF, [])), 24},
        {flist(maps:get(h3_adjacency_fft,     SF, [])), 16},
        {[float_val(maps:get(route_coherence,    SF, 0.0))],  1},
        {flist(maps:get(urban_rural_vector,   RE, [])),  8},
        {flist(maps:get(density_embedding,    RE, [])),  8},
        {[float_val(maps:get(infrastructure_bias, RE, 0.5))], 1},
        {[AC],                                                  1},
        {[0.0],                                                 1}   %% reserved
    ],

    Vec = lists:flatmap(fun({Vals, Budget}) ->
                            fit(Vals, Budget)
                        end, Segments),

    %% Safety: guarantee exactly 128 dims regardless of input
    fit(Vec, 128).

%% @doc Derive a stable non_neg_integer node ID from a JointSpectralEvent.
-spec event_to_node_id(map()) -> non_neg_integer().
event_to_node_id(Event) ->
    EID = maps:get(event_id, Event, <<>>),
    PT  = maps:get(phenomenon_time, Event, 0),
    erlang:phash2({EID, PT bsr 20}, 16#0FFFFFFF).

%%====================================================================
%% Internal
%%====================================================================

%% Build the metadata map for hann_grg
event_to_meta(Event) ->
    SF    = maps:get(geometric_fft,    Event, #{}),
    RE    = maps:get(region_embedding, Event, #{}),
    SMeta = maps:get(metadata, SF, #{}),
    #{
        h3_index             => maps:get(h3_index, SMeta, null),
        route_ids            => [],
        region               => null,
        alignment_confidence => maps:get(alignment_confidence, Event, 0.0),
        phenomenon_time      => maps:get(phenomenon_time,      Event, 0),
        infrastructure_bias  => maps:get(infrastructure_bias,  RE,    null),
        route_coherence      => maps:get(route_coherence,      SF,    null)
    }.

%% Truncate to N dims or pad with 0.0 to reach N dims
fit(List, N) ->
    Len = length(List),
    if
        Len >= N -> lists:sublist(List, N);
        true     -> List ++ lists:duplicate(N - Len, 0.0)
    end.

%% Ensure all elements are floats
flist(List) -> [float_val(X) || X <- List].

float_val(X) when is_float(X)   -> X;
float_val(X) when is_integer(X) -> float(X);
float_val(_)                    -> 0.0.
