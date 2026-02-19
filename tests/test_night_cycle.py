"""Tests for the Night Cycle maintenance runner."""

from continuity_core.memory.system import TieredMemorySystem
from continuity_core.services.night_cycle import NightCycle


def _make_system():
    return TieredMemorySystem()


def test_night_cycle_runs_without_graph():
    mem = _make_system()
    mem.remember("fact one", memory_type="semantic", importance=5)
    mem.event_log.log("user", "ask", "question?", "fact one is true")

    cycle = NightCycle(mem)
    result = cycle.run(graph=None)

    assert result.decay_applied is True
    assert result.duration_sec >= 0


def test_night_cycle_prunes_low_salience():
    mem = _make_system()
    # Add an item and set its salience very low
    item_id = mem.remember("ephemeral", memory_type="semantic", importance=1)
    for item in mem._fallback._items:
        if item.id == item_id:
            item.salience = 0.01

    cycle = NightCycle(mem, prune_threshold=0.05)
    result = cycle.run()

    assert result.items_pruned >= 1


def test_night_cycle_detects_contradictions():
    mem = _make_system()
    mem.event_log.log("user", "claim", "", "The system is fast and reliable")
    mem.event_log.log("user", "claim", "", "The system is not fast or reliable")

    cycle = NightCycle(mem)
    result = cycle.run()

    assert result.contradictions_found >= 1


def test_night_cycle_detects_resolutions():
    mem = _make_system()
    # First cycle: create a contradiction
    mem.event_log.log("user", "claim", "", "X is good")
    mem.event_log.log("user", "claim", "", "X is not good")

    cycle = NightCycle(mem)
    result1 = cycle.run()
    assert result1.contradictions_found >= 1

    # Now "resolve" by logging consistent statements and clearing old ones.
    # Simulate resolution: the next cycle will see different statements.
    # We force the event log to only have consistent statements.
    mem._event_log._store._events.clear()
    mem.event_log.log("user", "claim", "", "X is good")
    mem.event_log.log("user", "claim", "", "X is great")

    result2 = cycle.run()
    assert len(result2.resolutions) >= 1
    assert result2.resolutions[0]["type"] == "contradiction_resolved"


def test_night_cycle_with_graph():
    mem = _make_system()
    mem.event_log.log("user", "note", "", "concept A")

    graph = {
        "a": {"b"},
        "b": set(),
        "c": {"d"},
        "d": set(),
    }
    cycle = NightCycle(mem)
    result = cycle.run(graph=graph)

    assert result.voids_found >= 1


def test_night_cycle_logs_event():
    mem = _make_system()
    cycle = NightCycle(mem)
    cycle.run()

    events = mem.event_log.query(tag="maintenance")
    assert len(events) >= 1
    assert events[-1].input == "night_cycle"


def test_harmonic_integration_reward():
    mem = _make_system()
    mem.remember("helpful memory", memory_type="semantic", importance=7)

    # First cycle creates stress
    mem.event_log.log("user", "claim", "", "A is correct")
    mem.event_log.log("user", "claim", "", "A is not correct")
    cycle = NightCycle(mem)
    result1 = cycle.run()

    # Resolve the contradiction
    mem._event_log._store._events.clear()
    mem.event_log.log("user", "claim", "", "A is correct")

    result2 = cycle.run()
    # Reward should be non-negative (stress either dropped or stayed)
    assert result2.harmonic_reward >= 0.0
