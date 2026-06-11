# GuelminAsafi

**HNSW-backed spatiotemporal vector search for the YesBana / Matatu Pulse stack**
Erlang/OTP · ETS concurrent reads · ScyllaDB via marina · Cowboy HTTP

*Named after Guelmim and Safi — Moroccan trans-Saharan trade cities that served
as routing waypoints and dispatch ports for centuries before "routing" was a
software concept.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           Semantic Cognition Layer  (xAI / HRL)                 │
│   Input contract: regiondcl_embedding, grg_neighbors[]          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ POST /v1/grg/query
┌──────────────────────────▼──────────────────────────────────────┐
│                        GuelminAsafi                             │
│                                                                 │
│  hann_http_handler  ← Cowboy REST (11 routes)                  │
│  hann_grg           ← Cloverleaf GRG enrichment; meta from ETS │
│                                                                 │
│  hann_hnsw  (gen_server — serialised writes)                   │
│  │  add/2  → ETS write → async ScyllaDB persist                │
│  └─ search/2 ──────────────────────────────────────────────┐   │
│                                                            │   │
│  ┌──────────────────────────────────────────────────────┐  │   │
│  │  hann_ets  (ETS owner gen_server)                    │◄─┘   │
│  │                                                      │       │
│  │  hann_nodes       {ID, Vec, Layers}                  │       │
│  │  hann_idx_state   {entry, max_layer, ef, dist_fn}   │ ← search/2
│  │  hann_meta        {ID, MetaMap}                      │   direct ETS
│  │  public + read_concurrency: true                     │   no gen_server
│  └──────────────────────────┬───────────────────────────┘   hop
│                             │ async  (marina:query/2)
│  ┌──────────────────────────▼───────────────────────────┐       │
│  │  hann_scylla  (gen_server, ready-flag, retry loop)   │       │
│  │  ScyllaDB: himap.nodes / node_meta / hnsw_state      │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                 │
│  hann_vector  ← pure cosine / euclidean math                   │
│                                                                 │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │  local_stream_buffer  (hann_lsb)                          │   │
│ │                                                           │   │
│ │  Hypntyz Window   ordered_set {TS,Seq} + event index     │   │
│ │  Shek3m  Window   ordered_set {TS,Seq} + event index     │   │
│ │  Region Embedding gen_server state (slow, no FFT)        │   │
│ │                         │                                │   │
│ │            Alignment Engine                              │   │
│ │            Exact path  O(1)  — event-id hash lookup      │   │
│ │            Fuzzy path  O(k)  — sentinel seek + range     │   │
│ │                         │                                │   │
│ │            JointSpectralEvent                            │   │
│ │            (temporal_fft + geometric_fft +               │   │
│ │             region_embedding + alignment_confidence)     │   │
│ │                         │                                │   │
│ └─────────────────────────┼────────────────────────────────┘   │
│                           │ emit_fun callback                   │
│  hann_sink  ←─────────────┘                                     │
│  JointSpectralEvent → 128-d RegionDCL vector → hann_hnsw:add/2 │
└─────────────────────────────────────────────────────────────────┘
```

### Data flow — per matatu GPS tick

```
Hypntyz broadcast  ──►  POST /v1/stream/hypntyz
Shek3m  broadcast  ──►  POST /v1/stream/shek3m
Region  update     ──►  POST /v1/stream/region
                              │
                    local_stream_buffer
                    (alignment engine)
                              │
                          hann_sink
                    (vector assembly + node ID)
                              │
                         hann_hnsw:add/2
                         hann_grg:register_node_meta/2
                              │
                    POST /v1/grg/query
                              │
                    Semantic Cognition Layer
```

### Read / Write split

| Path | Mechanism | Concurrency |
|------|-----------|-------------|
| `search/2` | Direct ETS (`read_concurrency: true`) | N schedulers, zero serialisation |
| `add/2` | `gen_server:call` → ETS → async ScyllaDB | Serialised (HNSW invariants) |
| `query_neighbors/2` | `search/2` + ETS meta lookup | Fully concurrent |
| `ingest_hypntyz/2` | `gen_server:cast` to hann_lsb | Async, non-blocking |

### Supervisor strategy: `rest_for_one`

```
hann_ets            → crash restarts all (ETS tables gone)
hann_scylla         → crash restarts scylla + hnsw + grg + lsb
hann_hnsw           → crash restarts hnsw + grg + lsb (ETS intact)
hann_grg            → crash restarts grg + lsb
local_stream_buffer → crash restarts only lsb (windows lost, acceptable)
```

`hann_scylla` boots immediately even without ScyllaDB — it retries
connection every 5 s via `self() ! connect` in `init/1`.

---

## Modules

| Module | Role |
|--------|------|
| `hann_ets` | Owns 3 named ETS tables; all helpers are direct ETS calls |
| `hann_scylla` | marina wrapper — schema DDL, ETS hydration on boot, async writes |
| `hann_hnsw` | HNSW index gen_server; `search/2` bypasses it entirely |
| `hann_grg` | Cloverleaf GRG — enriches kNN results with H3 / stop metadata |
| `hann_vector` | Pure vector math — cosine, euclidean, dot, norm, normalize |
| `hann_http_handler` | Cowboy handler for all 11 REST routes |
| `local_stream_buffer` | Dual-stream spectral synchroniser with sliding windows |
| `hann_sink` | Converts JointSpectralEvents → 128-d vectors → hann_hnsw |

---

## LocalStreamBuffer — spectral alignment engine

`local_stream_buffer` synchronises two real-time FFT broadcast streams:

| Stream | Source | Content |
|--------|--------|---------|
| **Hypntyz** | Temporal FFT broadcast | speed_fft, accel_spectrum, motion_harmonics |
| **Shek3m** | Geometric FFT broadcast | corridor_fft, h3_adjacency_fft, route_coherence |

Region embedding is slow state held in the gen_server record — updated
explicitly, never FFT'd, acts as a cohesion bias on alignment confidence.

### Alignment confidence

```
confidence = 0.50 × event_id_match       (1.0 if same ID, else 0.0)
           + 0.30 × temporal_proximity   (1 - |dt| / delta_time)
           + 0.20 × infrastructure_bias  (from region embedding)
```

### Alignment complexity

| Path | Trigger | Complexity |
|------|---------|------------|
| Exact | Same `event_id` in both streams | O(1) — event index hash lookup |
| Fuzzy | No matching `event_id` | O(k) — sentinel seek + range walk |
| Early-exit | Newest exterior frame predates window | O(1) — `ets:last/1` check |

### Key design

Keys are `{Timestamp_µs, erlang:unique_integer([monotonic])}`.
Lower-bound seek uses `ets:next(Tab, {Lo - 1, []})` — the `[]` sentinel
is greater than any integer in Erlang term order, making this O(log n).

### hann_sink vector layout (128 dims)

```
Slot   0-31  (32)  speed_fft           HypntyzFrame
Slot  32-51  (20)  accel_spectrum      HypntyzFrame
Slot  52-67  (16)  motion_harmonics    HypntyzFrame
Slot  68-91  (24)  corridor_fft        Shek3mFrame
Slot  92-107 (16)  h3_adjacency_fft    Shek3mFrame
Slot  108     (1)  route_coherence     Shek3mFrame
Slot 109-116  (8)  urban_rural_vector  RegionEmbedding
Slot 117-124  (8)  density_embedding   RegionEmbedding
Slot  125     (1)  infrastructure_bias RegionEmbedding
Slot  126     (1)  alignment_confidence
Slot  127     (1)  reserved (0.0)
```

---

## Dependencies

| Dep | Source | Purpose |
|-----|--------|---------|
| `cowboy 2.10` | hex.pm | HTTP server |
| `jiffy 1.1` | hex.pm | JSON codec |
| `marina` | github.com/lpgauth/marina master | ScyllaDB CQL native-protocol client |

**marina API note:** this version exports `query/2` and `async_query/2`,
not the legacy `query/4`. The adapter is in `hann_scylla:marina_call/3` —
one function to update if the request format needs adjusting. Check the
actual signature with:
```bash
cat _build/default/lib/marina/src/marina.erl | grep -A8 "^query("
```

---

## Setup

### 1. ScyllaDB schema *(run once)*

```bash
cqlsh -f priv/schema.cql
```

Creates keyspace `himap` and tables `nodes`, `node_meta`, `hnsw_state`.
`IF NOT EXISTS` guards make it safe to re-run.

### 2. Config *(config/sys.config)*

```erlang
{marina, [
    {bootstrap_ips,    ["127.0.0.1"]},
    {keyspace,         "himap"},
    {port,             9042},
    {pool_size,        16},
    {requests_timeout, 5000}
]}
```

### 3. Build

```bash
rebar3 compile
rebar3 shell       # dev shell — boots without ScyllaDB (retries in background)
rebar3 eunit       # unit + integration tests (mock ScyllaDB, no live DB needed)
rebar3 release     # production OTP release
```

---

## HNSW Parameters *(config/sys.config)*

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hnsw_dim` | 128 | RegionDCL embedding dimension |
| `hnsw_m` | 16 | Max edges per node per layer |
| `hnsw_ef` | 64 | Beam width during construction |
| `hnsw_dist_fn` | `cosine` | `cosine` or `euclidean` |
| `port` | 8080 | HTTP listener port |

---

## REST API

### Health
```
GET /health
← 200  {"status":"ok","node_count":177312,"max_layer":6}
```

### Add / bulk-add nodes
```
POST /v1/node
{"id": 45012, "vector": [0.45, -0.12, …]}
← 201  {"status":"inserted","id":45012}

POST /v1/index/bulk
{"nodes": [{"id": 45012, "vector": […]}, …]}
← 200  {"status":"ok","inserted":2}
```

### kNN search
```
POST /v1/search
{"vector": [0.45, -0.12, …], "k": 5}
← 200  {"results":[{"node_id":45012,"distance":0.031}],"count":1}
```

### Cloverleaf GRG query *(primary Cognition Layer endpoint)*
```
POST /v1/grg/query
{"regiondcl_embedding": [0.45, -0.12, …], "k": 5}
← 200  {
    "grg_neighbors": [{
        "node_id": 45012, "distance": 0.031, "similarity": 0.984,
        "rank": 1, "h3_index": "8a182da6a4c7fff",
        "stop_name": "GPO Nairobi", "route_ids": [42,107], "region": "Nairobi CBD"
    }],
    "count": 5
  }

POST /v1/grg/meta
{"node_id": 45012, "meta": {"h3_index":"8a182da6a4c7fff","stop_name":"GPO Nairobi"}}
← 200  {"status":"registered","node_id":45012}
```

### Stream ingestion *(LocalStreamBuffer)*
```
POST /v1/stream/hypntyz
{
  "event_id": "evt-001",
  "phenomenon_time": 1749601275000000,
  "speed_fft": [0.1, 0.4, 0.9, 0.3],
  "accel_spectrum": [0.2, 0.1, 0.05],
  "motion_harmonics": [1.0, 0.5, 0.25],
  "metadata": {"vehicle_id": "NKB-001"}
}
← 200  {"status":"accepted"}

POST /v1/stream/shek3m
{
  "event_id": "evt-001",
  "phenomenon_time": 1749601275050000,
  "corridor_fft": [0.3, 0.8, 0.6],
  "h3_adjacency_fft": [0.5, 0.4, 0.2],
  "route_coherence": 0.91,
  "metadata": {"h3_index": "8a1e2459b4fffff"}
}
← 200  {"status":"accepted"}

POST /v1/stream/region
{
  "urban_rural_vector": [0.8, 0.2],
  "density_embedding": [0.9, 0.7, 0.6],
  "infrastructure_bias": 0.85,
  "updated_at": 1749601200000000
}
← 200  {"status":"ok"}

GET /v1/stream/results
← 200  {"events":[{…}],"count":1}

GET /v1/stream/counters
← 200  {"hypntyz":142,"shek3m":139,"emitted":97}
```

---

## Erlang API

```erlang
%% HNSW index
hann_hnsw:add(NodeID, Vector)     -> ok | {error, _}.
hann_hnsw:search(QueryVec, K)     -> [{Distance, NodeID}].
hann_hnsw:bulk_add([{ID, Vec}])   -> {ok, Count}.

%% GRG
hann_grg:query_neighbors(Emb, K)           -> [map()].
hann_grg:register_node_meta(NodeID, Meta)  -> ok.

%% LocalStreamBuffer (registered as hann_lsb)
local_stream_buffer:ingest_hypntyz(hann_lsb, Frame)  -> ok.
local_stream_buffer:ingest_shek3m(hann_lsb, Frame)   -> ok.
local_stream_buffer:update_region(hann_lsb, RE)       -> ok.
local_stream_buffer:fetch_results(hann_lsb)           -> [JointSpectralEvent].
local_stream_buffer:counters(hann_lsb)                -> map().
```

---

## Index selection

| Index | Chosen? | Reason |
|-------|---------|--------|
| **HNSW** | ✅ | O(log n) search, cosine support, dynamic inserts, no retraining |
| PQIVF | ❌ | Requires `Train()` — breaks on live traffic-pattern updates |
| RPT | ❌ | Euclidean only, O(n) worst-case on dense Nairobi H3 clusters |

---

## Integration with YesBana / Matatu Pulse

1. **Bootstrap**: run `cqlsh -f priv/schema.cql`, configure marina in
   `sys.config`. `hann_scylla` hydrates ETS from ScyllaDB on boot with
   automatic retry — service starts immediately even without a live DB.

2. **Region state**: load the current H3 regional embeddings once at startup
   via `POST /v1/stream/region`. Update when urban density or infrastructure
   data changes (not per-event).

3. **Per-event ingestion**: on each matatu GPS tick, post the Hypntyz temporal
   FFT frame and Shek3m geometric FFT frame to their respective endpoints.
   When a pair aligns above the confidence threshold, `hann_sink` assembles
   the 128-d RegionDCL vector and indexes it into the HNSW graph automatically.

4. **Routing query**: call `POST /v1/grg/query` with the current context
   vector. The response `grg_neighbors[]` populates the Cognition Layer
   input contract directly — node IDs, distances, H3 indices, route IDs.

5. **Dynamic updates**: surge, roadblock, detour → the stream alignment engine
   produces new JointSpectralEvents that update existing nodes or insert new
   ones. No retraining or restart required.

6. **Crash recovery**: `hann_hnsw` crash → restarts in milliseconds from ETS.
   `hann_ets` crash → full ScyllaDB reload (automatic, via retry loop).
   `local_stream_buffer` crash → fresh windows; buffered frames since last
   emit are lost but new frames resume immediately.
