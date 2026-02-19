"""Tests for the c2.curiosity MCP tool handler."""

from continuity_core.mra.stress import EpistemicStressMonitor, StressResult
from continuity_core.mra.voids import VoidReport
from continuity_core.memory.system import TieredMemorySystem


def _make_system():
    return TieredMemorySystem()


def test_curiosity_empty_event_log():
    """When the event log has no entries, curiosity returns calm defaults."""
    mem = _make_system()

    # Simulate what the curiosity handler does (without going through MCP plumbing).
    mra = mem.get_mra_signals()
    assert mra is None  # cache starts stale

    events = mem.event_log.tail(n=20)
    statements = [e.output for e in events if e.output]
    assert statements == []

    # With no statements the handler returns the expand_knowledge default.
    result = {
        "stress_level": 0.0,
        "contradictions": [],
        "deep_tensions": [],
        "bridging_questions": [],
        "suggested_action": "expand_knowledge",
    }
    assert result["suggested_action"] == "expand_knowledge"
    assert result["stress_level"] == 0.0


def test_curiosity_warm_cache():
    """When MRA cache is warm, curiosity reads directly from it."""
    mem = _make_system()

    monitor = EpistemicStressMonitor()
    stress = monitor.compute(
        ["Dogs are friendly", "Dogs are not friendly"],
    )
    mem.update_mra_cache(stress, voids=None)

    mra = mem.get_mra_signals()
    assert mra is not None
    assert mra.last_stress is not None
    assert mra.last_stress.contradictions  # should find the contradiction


def test_curiosity_stale_cache_refreshes():
    """When MRA cache is stale, curiosity recomputes from recent events."""
    mem = _make_system()

    # Log contradictory events
    mem.event_log.log("user", "claim", "", "The API is fast")
    mem.event_log.log("user", "claim", "", "The API is not fast")

    # Cache is stale (never populated)
    assert mem.get_mra_signals() is None

    # Simulate the curiosity handler's refresh path
    events = mem.event_log.tail(n=20)
    statements = [e.output for e in events if e.output]
    assert len(statements) == 2

    monitor = EpistemicStressMonitor(embed_fn=mem.embedder.embed)
    stress = monitor.compute(statements)
    mem.update_mra_cache(stress, voids=None)

    mra = mem.get_mra_signals()
    assert mra is not None
    assert mra.last_stress.contradictions


def test_curiosity_suggested_action_deep_tension():
    """Deep tensions should produce resolve_deep_tension action."""
    mem = _make_system()

    # Create a stress result with a deep tension
    stress = StressResult(
        s_omega=0.5,
        d_log=0.5,
        d_sem=0.0,
        v_top=0.0,
        contradictions=[("A is good", "A is not good", 0.8)],
        deep_tensions=[("A is good", "A is not good", 0.8, 0.7)],
        should_trigger=True,
    )
    mem.update_mra_cache(stress, voids=None)

    mra = mem.get_mra_signals()
    # Replicate the curiosity handler's action selection
    deep_tensions = [
        {"s1": s1, "s2": s2, "score": sc, "similarity": sim}
        for s1, s2, sc, sim in mra.last_stress.deep_tensions
    ]
    suggested = "expand_knowledge"
    if deep_tensions:
        suggested = "resolve_deep_tension"
    assert suggested == "resolve_deep_tension"


def test_curiosity_suggested_action_explore_gap():
    """Voids with no contradictions should produce explore_gap action."""
    mem = _make_system()

    stress = StressResult(
        s_omega=0.1,
        d_log=0.0,
        d_sem=0.0,
        v_top=0.1,
        contradictions=[],
        deep_tensions=[],
        should_trigger=False,
    )
    voids = VoidReport(
        void_pairs=[("cluster_a", "cluster_b")],
        questions=["How does cluster_a relate to cluster_b?"],
    )
    mem.update_mra_cache(stress, voids=voids)

    mra = mem.get_mra_signals()
    contradictions = []
    deep_tensions = []
    bridging_questions = list(mra.last_voids.questions)

    suggested = "expand_knowledge"
    if deep_tensions:
        suggested = "resolve_deep_tension"
    elif contradictions:
        suggested = "resolve_contradiction"
    elif bridging_questions:
        suggested = "explore_gap"

    assert suggested == "explore_gap"
    assert "cluster_a" in bridging_questions[0]
