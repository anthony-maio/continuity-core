# MRA Feedback Loop Implementation Plan

## Goal

Close the gap between C2's documented intent ("directed curiosity," "motivation") and
what it actually delivers. Today, MRA is fire-and-forget: stress is computed but never
fed back into context selection. Consolidation/decay exist as code but are never invoked.
The agent flying on C2 never feels epistemic tension or gets motivated to resolve gaps.

After this work, calling `c2.context` will automatically surface contradictions,
knowledge gaps, and bridging questions — making MRA a live "motivation" channel that
shapes what the pilot agent sees in its context window.

## Design Principles (adapted from the standalone MRA repo)

- **Three-channel stress scoring**: Logical dissonance (NLI/heuristic), semantic
  divergence (embedding-based), and topological sparsity (graph connectivity).
  Match the weighting model from the standalone MRA.
- **Deep tension detection**: When two statements are both highly contradictory AND
  semantically similar, that signals fundamental instability — not just noise.
  Port this multiplier concept.
- **Feedback, not autonomy**: Unlike the standalone MRA which drives its own LLM to
  interrogate concepts, C2's MRA feeds signals into the pilot's context window.
  The agent decides what to do with tensions. We don't add LLM calls inside C2.
- **No new heavy dependencies**: No LangGraph, vLLM, FAISS, or DeBERTa in C2.
  Use the embedder already wired (hash/sbert/ollama) for semantic divergence.
  Support pluggable NLI via the existing `NliFn` protocol.
- **Tests for every change**: Each step adds or extends tests before or alongside code.

---

## Phase 1: Wire MRA Into Reality

### Step 1 — Deep tension multiplier in EpistemicStressMonitor

**Files:** `continuity_core/mra/stress.py`, `tests/test_mra_stress.py`

Port the "deep tension" concept from the standalone MRA: when two statements have
both high contradiction (>threshold) AND high semantic similarity (>threshold),
apply a multiplier (1.5x). This detects *fundamental* instabilities, not just surface
disagreements.

- Add `deep_tension_threshold` and `deep_tension_multiplier` config to `__init__`.
- In `_logical_dissonance`, after computing the base score for a pair, also check
  semantic similarity (requires embed_fn). If contradiction > 0.7 and similarity > 0.6,
  multiply the pair score by the deep tension multiplier.
- Add a `deep_tensions` list to `StressResult` tracking these pairs.
- Add `tests/test_mra_stress.py` with cases for: basic dissonance, deep tension
  amplification, semantic divergence with embedder, and the no-embedder fallback path.

### Step 2 — Wire embedder and graph sparsity into introspect tool

**Files:** `continuity_core/mcp/tools/introspect.py`, `continuity_core/services/runtime.py`

Currently `EpistemicStressMonitor()` is created with no `embed_fn` and no `nli_fn`,
and `graph_sparsity` is hardcoded to 0.0. Fix this:

- In `introspect()`, get the memory system's embedder from `get_memory_system().embedder`
  and pass `embed_fn=embedder.embed` to `EpistemicStressMonitor`.
- Compute actual `graph_sparsity` from the provided graph dict: ratio of actual edges
  to possible edges in the graph (density). If no graph, default to 1.0 (maximally sparse
  = maximally stressed about topology).
- Add a `_graph_sparsity()` helper function.
- Add `deep_tensions` to the returned stress dict.
- Test: `tests/test_introspect.py` — verify embed_fn is wired, verify sparsity computation.

### Step 3 — MRA stress cache on TieredMemorySystem

**Files:** `continuity_core/memory/system.py`

The memory system needs to hold the latest MRA results so the context pipeline can use
them without re-computing on every call.

- Add a `MRACache` dataclass: `last_stress: Optional[StressResult]`,
  `last_voids: Optional[VoidReport]`, `updated_at: float`, `staleness_sec: float = 300`.
- Add `self._mra_cache: MRACache` to `TieredMemorySystem.__init__`.
- Add `update_mra_cache(stress, voids)` and `get_mra_signals()` methods.
  `get_mra_signals()` returns None if cache is stale (older than staleness_sec).
- Test: `tests/test_mra_cache.py` — cache round-trip, staleness expiry.

---

## Phase 2: MRA Signals Flow Into Context

### Step 4 — Inject MRA signals into CandidateGatherer

**Files:** `continuity_core/context/gather.py`

When gathering candidates, if there are cached MRA signals, inject them as synthetic
high-priority candidates so the agent *sees* the tensions.

- After gathering from all four sources (working, vector, graph, events), check
  `self._memory.get_mra_signals()`.
- If stress `should_trigger` is True:
  - For each contradiction in `contradictions`, create a candidate with store="mra",
    text="[Contradiction] {stmt1} vs. {stmt2} (score: {score})",
    high relevance (0.95), high salience (0.9).
  - For each bridging question in voids, create a candidate with store="mra",
    text="[Knowledge Gap] {question}", relevance=0.85, salience=0.85.
  - For deep tensions specifically, create candidates with even higher salience (0.95).
- Cap MRA injections at a configurable max (default: 5) to avoid flooding.
- Test: `tests/test_gather_mra.py` — verify MRA candidates appear when cache is hot,
  don't appear when cache is stale or empty.

### Step 5 — New MCP tool: `c2.curiosity`

**Files:** `continuity_core/mcp/tools/curiosity.py`, `continuity_core/mcp/server.py`

A dedicated tool for the agent to ask "What should I be curious about?" This is the
pull-based complement to the push-based injection in Step 4.

- `curiosity(arguments)`:
  - Get memory system and MRA cache.
  - If cache is stale or empty, run a fresh introspect cycle using recent statements
    from the event log (last N events as statements).
  - Return: `{ stress_level, contradictions[], deep_tensions[], bridging_questions[],
    suggested_action }`.
  - `suggested_action` is a heuristic: if deep tensions exist → "resolve_contradiction",
    if voids exist → "explore_gap", if stress is low → "expand_knowledge".
- Register in `server.py` as `c2.curiosity`.
- Test: `tests/test_curiosity_tool.py`.

---

## Phase 3: Activate Consolidation and Decay

### Step 6 — Decay on recall

**Files:** `continuity_core/memory/system.py`

- Add a `_recall_count: int` counter.
- Every 10th `recall()` call, run `self._fallback.apply_decay(rate, time_unit)` if
  using fallback, or write a decay sweep event if using Qdrant (update salience via
  payload update). This is a lightweight lifecycle trigger.
- Test: extend `tests/test_consolidation.py` — verify decay triggers after N recalls.

### Step 7 — Credit assignment via feedback events

**Files:** `continuity_core/memory/system.py`, `continuity_core/mcp/tools/events.py`

- Add `credit(memory_ids: List[str], signal: float)` method to `TieredMemorySystem`.
  For each id, boost salience by `signal` (clamped to [0, 1]). If Qdrant, update the
  payload. If fallback store, update the MemoryItem directly.
- In `write_event`, when `intent == "credit"`, parse `metadata["memory_ids"]` and
  `metadata["signal"]` and call `self.credit()`.
- Test: `tests/test_credit.py` — verify salience boost on in-memory store.

### Step 8 — Wire RecallGatedConsolidator into recall path

**Files:** `continuity_core/memory/system.py`

- Instantiate `RecallGatedConsolidator` in `TieredMemorySystem.__init__`.
- After `recall()` returns results, for each result compute
  `should_consolidate(recall_strength=score, occupancy=len(items)/capacity)`.
  If True, boost the item's access metadata (touch). If False and score is very low,
  mark for future pruning.
- This activates the gating logic that's currently orphaned.
- Test: extend consolidation tests — verify gate decisions affect item lifecycle.

---

## Phase 4: Dynamic Scoring Dimensions

### Step 9 — Centrality from graph topology

**Files:** `continuity_core/context/gather.py`

- In `_graph_candidates()`, when querying Neo4j nodes, also fetch their degree count
  (number of edges). Normalize to [0, 1] and use as `centrality` instead of the
  hardcoded 0.6.
- For non-graph candidates, leave centrality at default but make it derivable
  if the content matches a known graph node.
- Test: `tests/test_gather_centrality.py`.

### Step 10 — Confidence from MRA contradiction status

**Files:** `continuity_core/context/gather.py`

- After gathering all candidates, cross-reference with MRA cache. If a candidate's
  text appears in a contradiction pair, reduce its `confidence` proportionally to the
  contradiction score. Contradicted items are less trustworthy.
- Test: extend gather tests.

---

## Execution Order and Dependencies

```
Step 1 (stress.py) ──────────┐
                              ├──→ Step 2 (introspect wiring)
Step 3 (MRA cache) ──────────┤
                              ├──→ Step 4 (gather injection)
                              │
                              └──→ Step 5 (c2.curiosity tool)

Step 6 (decay on recall) ────┐
Step 7 (credit assignment) ──┼──→ Step 8 (consolidator wiring)
                              │
Step 9 (centrality) ──────────┘
Step 10 (confidence from MRA) ── depends on Steps 3-4
```

Steps 1, 3, 6, 7, 9 can be done in parallel.
Steps 2, 4, 5 depend on 1 and 3.
Step 8 depends on 6 and 7.
Step 10 depends on 3 and 4.

---

## What This Does NOT Include

- **LLM-driven interrogation inside C2.** The standalone MRA repo has its own LLM loop
  for generating perspectives. C2 intentionally does NOT do this — the pilot agent is
  the LLM. C2 surfaces signals; the agent acts on them.
- **TTT adaptive scoring layer.** The standalone MRA has a test-time-training weight
  adaptation layer. This is a powerful idea but depends on utility feedback signals that
  C2 doesn't yet generate. Recommend as a follow-up after credit assignment (Step 7) is
  producing real signal.
- **LangGraph orchestration.** C2 stays with its own simple pipeline architecture.
- **New heavy dependencies.** No DeBERTa, no FAISS, no vLLM. The NliFn protocol already
  supports plugging in external NLI if desired.
