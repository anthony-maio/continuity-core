# Architecture

Continuity Core (C2) is organized around an append-only Event Log and a set of specialized memory stores. The Event Log is the source of truth; all other stores are indexes or projections and can be rebuilt.

## Core Loop

1) Ingest input into the Event Log.
2) Retrieve candidates from memory stores.
3) Compose a prompt pack with an expected-utility budget.
4) Execute reasoning and tools.
5) Assign credit to helpful memories.
6) Consolidate and decay under memory pressure.

## Biomimetic Memory Features

- Nested learning: consolidation rules adapt to retrieval accuracy and coherence gains.
- Gated plasticity: recall-gated consolidation with adaptive thresholds.
- Strategic forgetting: decay, pruning, and compression under pressure.

## MRA Integration

- Dissonance: contradictions + semantic divergence across perspectives.
- Voids: weakly connected clusters in the knowledge graph.
- Output: prioritized bridging inquiries.

## Store Roles

- L1 Working: Redis cache for recent context and short-term buffers.
- L2 Long-term: Qdrant vector memory for episodic and semantic items.
- L3 Archive: Postgres event log and checkpoints.
- Semantic KG: Neo4j for relationships, contradictions, and topology.

## MCP Server

The MCP server exposes a minimal tool surface for pilot mode:
- c2.write_event
- c2.context
- c2.introspect
