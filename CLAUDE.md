# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Continuity Core (C2) is a local-first cognitive memory engine and MCP server. It implements a biomimetic, multi-store memory architecture with the Manifold Resonance Architecture (MRA) for directed curiosity.

## Build & Development Commands

```bash
# Start infrastructure dependencies
docker compose up -d

# Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with ML dependencies (sentence-transformers)
pip install -e ".[ml]"

# Run the MCP server (stdio)
python -m continuity_core.mcp.server

# Ingest local documents
c2-ingest . --ext .md .txt

# Run all tests
pytest

# Run a single test file
pytest tests/test_event_log.py

# Run a specific test
pytest tests/test_event_log.py::test_event_log_append_and_query
```

## Architecture

### Data Flow
1. Ingest → Event Log (append-only source of truth)
2. Event Log → Memory stores (projections/indexes that can be rebuilt)
3. Retrieval → Candidate gathering from memory stores
4. Context Composer → Expected-utility selection with token budget
5. MRA → Background auditing for contradictions and knowledge gaps

### Storage Tiers
- **L1 Working (Redis)**: Recent context and short-term buffers
- **L2 Long-term (Qdrant)**: Vector memory for episodic and semantic items
- **L3 Archive (Postgres)**: Event log and checkpoints
- **Semantic KG (Neo4j)**: Relationships, contradictions, and topology

### Core Components

**Event Log** (`continuity_core/event_log.py`): Append-only journal. All other stores derive from this.

**TieredMemorySystem** (`continuity_core/memory/system.py`): Unified interface to all storage tiers. Handles graceful fallback when services are unavailable.

**ContextPipeline** (`continuity_core/context/pipeline.py`): Orchestrates candidate gathering and context composition.

**ContextComposer** (`continuity_core/context/composer.py`): Expected-utility ranker with epsilon-greedy exploration. Selects items within token budget.

**MRA Stress Monitor** (`continuity_core/mra/stress.py`): Computes epistemic stress from logical contradictions, semantic divergence, and topological sparsity.

**VoidDetector** (`continuity_core/mra/voids.py`): Identifies weakly-connected clusters in the knowledge graph and generates bridging questions.

### MCP Tools
- `c2.write_event`: Append to Event Log
- `c2.context`: Return context pack from ContextComposer
- `c2.introspect`: Run MRA stress/void scans

## Configuration

All settings via environment variables with `C2_` prefix. Key ones:
- `C2_REDIS_URL`, `C2_QDRANT_URL`, `C2_POSTGRES_URL`, `C2_NEO4J_URI`
- `C2_TOKEN_BUDGET`: Context composer token limit (default: 2048)
- `C2_EMBEDDING_BACKEND`: "hash" (default) or "sbert" or "ollama"
- `C2_DECAY_RATE`, `C2_RECENCY_HALF_LIFE_DAYS`: Memory decay parameters

## Testing Notes

Tests run without external dependencies by using in-memory fallbacks:
- `InMemoryEventStore` for Event Log
- `InMemoryStore` for vector memory
- TieredMemorySystem gracefully degrades when Redis/Qdrant/Neo4j unavailable

## Directory Structure

```
continuity_core/
  config.py           # C2Config dataclass, env var loading
  event_log.py        # Event, EventLog, EventStore protocol
  context/            # ContextComposer, CandidateGatherer, ContextPipeline
  memory/             # TieredMemorySystem, stores, decay, consolidation
  mra/                # EpistemicStressMonitor, VoidDetector
  mcp/                # MCP server and tool handlers
  storage/            # Redis, Qdrant, Postgres, Neo4j clients
  ingest/             # Document loaders, chunker, CLI
  graph/              # Canonical schema for knowledge graph
```
