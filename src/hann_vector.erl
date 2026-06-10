%%%-------------------------------------------------------------------
%%% @doc HiMap HANN — Vector Mathematics
%%%
%%% Pure functional module for all distance and similarity
%%% computations used by the HNSW index.
%%%
%%% Vectors are represented as [float()].  For 128-d RegionDCL
%%% embeddings this is compact enough; for extreme throughput
%%% you could swap to a binary/float32 NIFs later.
%%%-------------------------------------------------------------------
-module(hann_vector).

-export([
    distance/3,
    cosine/2,
    euclidean/2,
    squared_euclidean/2,
    dot/2,
    norm/1,
    normalize/1,
    zero/1,
    random_unit/1
]).

%%%-------------------------------------------------------------------
%%% Public API
%%%-------------------------------------------------------------------

%% @doc Unified distance dispatch.
-spec distance([float()], [float()], cosine | euclidean) -> float().
distance(A, B, cosine)    -> cosine(A, B);
distance(A, B, euclidean) -> euclidean(A, B).

%% @doc Cosine distance ∈ [0, 2].
%%      Returns 1 - cosine_similarity, so 0 = identical, 2 = opposite.
-spec cosine([float()], [float()]) -> float().
cosine(A, B) ->
    Dot  = dot(A, B),
    NA   = norm(A),
    NB   = norm(B),
    Denom = NA * NB,
    if
        Denom < 1.0e-12 -> 1.0;   %% treat zero-vectors as maximally distant
        true            -> 1.0 - (Dot / Denom)
    end.

%% @doc L2 Euclidean distance.
-spec euclidean([float()], [float()]) -> float().
euclidean(A, B) ->
    math:sqrt(squared_euclidean(A, B)).

%% @doc Squared Euclidean distance (cheaper, avoids sqrt for ranking).
-spec squared_euclidean([float()], [float()]) -> float().
squared_euclidean(A, B) ->
    zip_fold(fun(X, Y, Acc) -> Acc + (X - Y) * (X - Y) end, 0.0, A, B).

%% @doc Inner (dot) product.
-spec dot([float()], [float()]) -> float().
dot(A, B) ->
    zip_fold(fun(X, Y, Acc) -> Acc + X * Y end, 0.0, A, B).

%% @doc L2 norm (magnitude).
-spec norm([float()]) -> float().
norm(V) ->
    math:sqrt(lists:foldl(fun(X, Acc) -> Acc + X * X end, 0.0, V)).

%% @doc Unit-normalise a vector.  Returns V unchanged if ‖V‖ ≈ 0.
-spec normalize([float()]) -> [float()].
normalize(V) ->
    N = norm(V),
    if
        N < 1.0e-12 -> V;
        true        -> [X / N || X <- V]
    end.

%% @doc Zero vector of dimension D.
-spec zero(pos_integer()) -> [float()].
zero(D) -> lists:duplicate(D, 0.0).

%% @doc Random unit vector of dimension D (useful for testing).
-spec random_unit(pos_integer()) -> [float()].
random_unit(D) ->
    V = [rand:normal() || _ <- lists:seq(1, D)],
    normalize(V).

%%%-------------------------------------------------------------------
%%% Internal helpers
%%%-------------------------------------------------------------------

zip_fold(_Fun, Acc, [], []) ->
    Acc;
zip_fold(Fun, Acc, [H1 | T1], [H2 | T2]) ->
    zip_fold(Fun, Fun(H1, H2, Acc), T1, T2).
