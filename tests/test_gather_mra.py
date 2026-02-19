"""Tests for MRA signal injection into CandidateGatherer."""

import time
from continuity_core.context.gather import CandidateGatherer
from continuity_core.event_log import EventLog
from continuity_core.memory.system import MRACache, ScoredMemory
from continuity_core.mra.stress import StressResult
from continuity_core.mra.voids import VoidReport


class StubMemorySystem:
    """Lightweight stub — no real backends."""

    def __init__(self):
        self.event_log = EventLog()
        self.neo4j = None
        self._mra_cache = None
        self._memories = []

    def get_working_context(self, thread_id, limit=20):
        return []

    def recall(self, query, top_k=5, type_filter=None):
        return self._memories

    def get_mra_signals(self):
        return self._mra_cache

    def update_mra_cache(self, stress, voids=None):
        self._mra_cache = MRACache(
            last_stress=stress,
            last_voids=voids,
            updated_at=time.time(),
        )


def test_mra_candidates_injected_when_triggered():
    mem = StubMemorySystem()
    stress = StressResult(
        s_omega=0.6, d_log=0.5, d_sem=0.1, v_top=0.0,
        contradictions=[("A is true", "A is false", 0.8)],
        deep_tensions=[("X works well", "X fails badly", 0.95, 0.7)],
        should_trigger=True,
    )
    voids = VoidReport(
        void_pairs=[],
        questions=["How does concept Y relate to concept Z?"],
    )
    mem.update_mra_cache(stress, voids)

    gatherer = CandidateGatherer(mem)
    candidates, _ = gatherer.gather("test query")

    mra_candidates = [c for c in candidates if c.store == "mra"]
    assert len(mra_candidates) >= 2  # deep tension + bridging question

    texts = [c.text for c in mra_candidates]
    assert any("[Deep Tension]" in t for t in texts)
    assert any("[Knowledge Gap]" in t for t in texts)


def test_mra_candidates_not_injected_when_not_triggered():
    mem = StubMemorySystem()
    stress = StressResult(
        s_omega=0.1, d_log=0.05, d_sem=0.05, v_top=0.0,
        should_trigger=False,
    )
    mem.update_mra_cache(stress)

    gatherer = CandidateGatherer(mem)
    candidates, _ = gatherer.gather("test query")

    mra_candidates = [c for c in candidates if c.store == "mra"]
    assert len(mra_candidates) == 0


def test_mra_candidates_not_injected_when_cache_stale():
    mem = StubMemorySystem()
    # Don't update cache — it stays None

    gatherer = CandidateGatherer(mem)
    candidates, _ = gatherer.gather("test query")

    mra_candidates = [c for c in candidates if c.store == "mra"]
    assert len(mra_candidates) == 0


def test_mra_injection_capped_at_max():
    mem = StubMemorySystem()
    stress = StressResult(
        s_omega=0.8, d_log=0.7, d_sem=0.1, v_top=0.0,
        contradictions=[
            (f"stmt {i} is true", f"stmt {i} is false", 0.8)
            for i in range(10)
        ],
        should_trigger=True,
    )
    mem.update_mra_cache(stress)

    gatherer = CandidateGatherer(mem, max_mra_injections=3)
    candidates, _ = gatherer.gather("test query")

    mra_candidates = [c for c in candidates if c.store == "mra"]
    assert len(mra_candidates) <= 3


def test_confidence_reduced_for_contradicted_candidates():
    mem = StubMemorySystem()
    mem._memories = [
        ScoredMemory(
            id="mem1", score=0.9, content="A is true and reliable",
            memory_type="semantic", payload={"importance": 8, "confidence": 0.8},
        ),
    ]

    stress = StressResult(
        s_omega=0.6, d_log=0.5, d_sem=0.1, v_top=0.0,
        contradictions=[("A is true", "A is false", 0.9)],
        should_trigger=True,
    )
    mem.update_mra_cache(stress)

    gatherer = CandidateGatherer(mem)
    candidates, _ = gatherer.gather("A is true")

    # Find the memory candidate (not the MRA signal)
    memory_cands = [c for c in candidates if c.store != "mra" and "A is true" in c.text]
    assert len(memory_cands) >= 1
    # Confidence should have been reduced from 0.8
    assert memory_cands[0].confidence < 0.8


def test_deep_tensions_prioritized_over_regular_contradictions():
    mem = StubMemorySystem()
    stress = StressResult(
        s_omega=0.8, d_log=0.7, d_sem=0.1, v_top=0.0,
        contradictions=[("X works well", "X fails badly", 0.95), ("B is fast", "B is slow", 0.6)],
        deep_tensions=[("X works well", "X fails badly", 0.95, 0.7)],
        should_trigger=True,
    )
    mem.update_mra_cache(stress)

    gatherer = CandidateGatherer(mem, max_mra_injections=5)
    candidates, _ = gatherer.gather("test")

    mra_candidates = [c for c in candidates if c.store == "mra"]
    deep_texts = [c.text for c in mra_candidates if "[Deep Tension]" in c.text]
    contradiction_texts = [c.text for c in mra_candidates if "[Contradiction]" in c.text]

    assert len(deep_texts) == 1
    # The "X works well" vs "X fails badly" pair should not be duplicated as a regular contradiction
    assert not any("X works well" in t for t in contradiction_texts)
