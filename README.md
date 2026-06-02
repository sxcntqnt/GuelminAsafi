# himap_hann

**HNSW-backed vector search for HiMap's spatiotemporal routing stack**  
Erlang/OTP · ETS concurrent reads · ScyllaDB via marina · Cowboy HTTP

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│          Semantic Cognition Layer  (xAI / HRL)            │
│  Input contract: regiondcl_embedding, grg_neighbors[]     │
└────────────────────────────┬──────────────────────────────┘
                             │ POST /v1/grg/query
┌────────────────────────────▼──────────────────────────────┐
│                       himap_hann                          │
│                                                           │
│  hann_http_handler  ← Cowboy REST (6 routes)             │
│  hann_grg           ← GRG enrichment; meta from ETS      │
│                                                           │
│  hann_hnsw (gen_server — serialised writes)              │
│  │  add/2   → ETS write → async ScyllaDB persist         │
│  └─ search/2 → ─────────────────────────────────────┐    │
│                                                      │    │
│  ┌───────────────────────────────────────────────┐   │    │
│  │  hann_ets  (ETS owner gen_server)             │◄──┘    │
│  │                                               │        │
│  │  hann_nodes      {ID, Vec, Layers}            │        │
│  │  hann_idx_state  {entry, max_layer, ef, …}   │ ← search/2
│  │  hann_meta       {ID, MetaMap}                │   reads here
│  │                                               │   directly,
│  │  public + read_concurrency: true              │   no gen_server
│  └───────────────────────┬───────────────────────┘   hop at all
│                          │ async  (lpgauth/marina)
│  ┌───────────────────────▼───────────────────────┐        │
│  │  hann_scylla  (gen_server)                    │        │
│  │  marina pool → ScyllaDB CQL native protocol   │        │
│  │  himap.nodes / node_meta / hnsw_state         │        │
│  └───────────────────────────────────────────────┘        │
│                                                           │
│  hann_vector  ← pure cosine / euclidean math              │
└───────────────────────────────────────────────────────────┘
```

### Read / Write split

| Path | Mechanism | Concurrency |
|------|-----------|-------------|
| `search/2` | Direct ETS (`read_concurrency: true`) | N schedulers in parallel |
| `add/2` | `gen_server:call` → ETS → async ScyllaDB | Serialised (HNSW invariants) |
| `query_neighbors/2` | Calls `search/2` + ETS meta lookup | Fully concurrent |

### Supervisor strategy: `rest_for_one`

```
hann_ets     → crash restarts everything (tables gone)
hann_scylla  → crash restarts scylla + hnsw + grg (DB reload)
hann_hnsw    → crash restarts only hnsw + grg (ETS intact)
hann_grg     → crash restarts only grg
```

---

## Modules

| Module | Role |
|--------|------|
| `hann_ets` | Owns the 3 named ETS tables; exposes direct-access helpers |
| `hann_scylla` | marina wrapper — DDL, startup hydration, async writes |
| `hann_hnsw` | HNSW index gen_server; `search/2` bypasses it entirely |
| `hann_grg` | Cloverleaf GRG — enriches kNN results with H3 / stop metadata |
| `hann_vector` | Pure vector math — cosine, euclidean, dot, norm, normalize |
| `hann_http_handler` | Cowboy handler for all 6 REST routes |

---

## Dependencies

| Dep | Source | Purpose |
|-----|--------|---------|
| `cowboy 2.10` | hex.pm | HTTP server |
| `jiffy 1.1` | hex.pm | JSON codec |
| `marina` | github.com/lpgauth/marina | ScyllaDB CQL native-protocol client |

---

## Setup

### 1. ScyllaDB schema  *(run once)*

```bash
cqlsh -f priv/schema.cql
```

Creates keyspace `himap` and tables `nodes`, `node_meta`, `hnsw_state`
with `IF NOT EXISTS` guards — safe to re-run.

### 2. marina config  *(config/sys.config)*

```erlang
{marina, [
    {bootstrap_ips,    ["127.0.0.1"]},
    {keyspace,         "himap"},
    {port,             9042},
    {pool_size,        16},
    {requests_timeout, 5000}
]}
```

### 3. Build and run

```bash
rebar3 compile
rebar3 shell           # development shell with sys.config loaded
rebar3 eunit           # tests (mock ScyllaDB — no live DB needed)
rebar3 release         # production OTP release
```

---

## HNSW Parameters  *(config/sys.config)*

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hnsw_dim` | 128 | RegionDCL embedding dimension |
| `hnsw_m` | 16 | Max edges per node per layer. Higher = better recall, more RAM |
| `hnsw_ef` | 64 | Beam width during construction. Higher = better graph quality |
| `hnsw_dist_fn` | `cosine` | `cosine` or `euclidean` |
| `port` | 8080 | HTTP listener port |

---

## REST API

### Health

```
GET /health
← 200  {"status":"ok","node_count":177312,"max_layer":6}
```

### Add a node

```
POST /v1/node
{"id": 45012, "vector": [0.45, -0.12, …]}   ← 128 floats

← 201  {"status":"inserted","id":45012}
```

### Bulk add  *(Africa OSM load)*

```
POST /v1/index/bulk
{"nodes": [{"id": 45012, "vector": […]}, …]}

← 200  {"status":"ok","inserted":2}
```

### Raw kNN search

```
POST /v1/search
{"vector": [0.45, -0.12, …], "k": 5}

← 200  {"results":[{"node_id":45012,"distance":0.031},…],"count":2}
```

### Cloverleaf GRG query  *(primary Cognition Layer endpoint)*

```
POST /v1/grg/query
{"regiondcl_embedding": [0.45, -0.12, …], "k": 5}

← 200  {
    "grg_neighbors": [{
        "node_id":    45012,
        "distance":   0.031,
        "similarity": 0.984,
        "rank":       1,
        "h3_index":   "8a182da6a4c7fff",
        "stop_name":  "GPO Nairobi",
        "route_ids":  [42, 107],
        "region":     "Nairobi CBD"
    }],
    "count": 5
  }
```

### Register node metadata

```
POST /v1/grg/meta
{"node_id": 45012, "meta": {"h3_index":"8a182da6a4c7fff","stop_name":"GPO Nairobi","route_ids":[42,107],"region":"Nairobi CBD"}}

← 200  {"status":"registered","node_id":45012}
```

---

## Erlang API

```erlang
%% Insert (serialised write → ETS + async ScyllaDB)
hann_hnsw:add(NodeID :: integer(), Vector :: [float()]) -> ok | {error, _}.

%% kNN search — direct ETS read, fully concurrent, no gen_server hop
hann_hnsw:search(QueryVec :: [float()], K :: integer()) -> [{float(), integer()}].

%% Bulk insert
hann_hnsw:bulk_add([{NodeID, Vector}]) -> {ok, Count}.

%% GRG query — concurrent search + ETS meta enrichment
hann_grg:query_neighbors(Embedding :: [float()], K :: integer()) -> [map()].

%% Register H3 / stop metadata (ETS + async ScyllaDB)
hann_grg:register_node_meta(NodeID :: integer(), Meta :: map()) -> ok.
```

---

## Index selection

| Index | Chosen? | Reason |
|-------|---------|--------|
| **HNSW** | ✅ | O(log n) search, cosine support, dynamic inserts safe, no retraining |
| PQIVF | ❌ | Requires `Train()` — breaks on live traffic-pattern updates |
| RPT | ❌ | Euclidean only, worst-case O(n) on dense urban H3 hex clusters |

---

## Integration with YesBana / Matatu Pulse

1. **Bootstrap**: run `cqlsh -f priv/schema.cql`, configure marina in `sys.config`,
   then start the service. `hann_scylla` hydrates ETS from ScyllaDB on boot —
   no separate warm-up step needed.

2. **Bulk load**: scan the Africa OSM DuckDB Parquet store and call
   `POST /v1/index/bulk` with precomputed RegionDCL embeddings.
   Writes land in ETS immediately and drain to ScyllaDB asynchronously.

3. **Per-event routing**: on each matatu GPS tick, compute the 128-d context
   vector and call `POST /v1/grg/query`. The response populates
   `grg_neighbors[]` in the Cognition Layer input contract directly.

4. **Dynamic updates**: surge, roadblock, or detour → call `add/2` with an
   updated embedding. HNSW handles live inserts without retraining.
   Modified neighbour lists are persisted to ScyllaDB automatically.

5. **Crash recovery**: if `hann_hnsw` crashes alone, the supervisor restarts it
   in milliseconds — it reads `entry` and `max_layer` back from ETS (still alive,
   owned by `hann_ets`) and is immediately ready. Only a full `hann_ets` crash
   triggers a ScyllaDB reload.
