# Continuity Core (C2)

Continuity Core is a local-first cognitive memory engine and MCP server. It unifies a biomimetic, multi-store memory architecture with the Manifold Resonance Architecture (MRA) for directed curiosity.

This repository is a fresh implementation that focuses on:
- Nested learning and gated plasticity for consolidation
- Strategic forgetting via decay and pruning
- A tiered memory model (working, episodic, semantic, procedural, autobiographical, goals)
- An append-only Event Log as the source of truth
- MRA stress and void detection for targeted inquiry

## Core Concepts

- Event Log: Append-only journal of all interactions and internal transitions.
- Memory Stores: Specialized stores with distinct decay and access patterns.
- Context Composer: Budgeted expected-utility selection of prompt context.
- MRA: Multi-perspective dissonance scoring + conceptual void detection.
- Credit Assignment: Reinforces memories that improve outcomes.

## MVP MCP Tools

- c2.write_event: Append to the Event Log and trigger indexing.
- c2.context: Return a prompt pack from the Context Composer.
- c2.introspect: Run MRA stress/void scans and return inquiry candidates.

## Quickstart (Local)

1) Start dependencies:

```bash
docker compose up -d
```

2) Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3) Install the package (editable):

```bash
pip install -e .
```

4) Run the MCP server (stdio):

```bash
python -m continuity_core.mcp.server
```

5) Ingest local notes (optional):

```bash
c2-ingest . --ext .md .txt
```

## Project Layout

```
continuity_core/
  config.py
  event_log.py
  context/
  graph/
  ingest/
  memory/
  mra/
  mcp/
  services/
```

## License

MIT. See LICENSE.
