-- HiMap HANN — ScyllaDB Schema
-- Run once with: cqlsh -f priv/schema.cql

CREATE KEYSPACE IF NOT EXISTS himap
  WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
  AND  durable_writes = true;

-- ── Node embeddings + HNSW adjacency lists ───────────────────────────
-- vector : float32 LE packed binary  (128 * 4 = 512 bytes)
-- layers : term_to_binary(#{Layer => [NodeID]})
CREATE TABLE IF NOT EXISTS himap.nodes (
    node_id  bigint  PRIMARY KEY,
    vector   blob,
    layers   blob
);

-- ── GRG node metadata (H3, stop names, route IDs) ───────────────────
CREATE TABLE IF NOT EXISTS himap.node_meta (
    node_id   bigint  PRIMARY KEY,
    h3_index  text,
    stop_name text,
    route_ids blob,   -- term_to_binary([int()])
    region    text
);

-- ── Single-row HNSW global state (entry-point + layer ceiling) ───────
CREATE TABLE IF NOT EXISTS himap.hnsw_state (
    key       text    PRIMARY KEY,
    entry     bigint,
    max_layer int
);
