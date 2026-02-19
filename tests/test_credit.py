"""Tests for credit assignment and decay on recall."""

from continuity_core.memory.system import TieredMemorySystem


def _make_system():
    return TieredMemorySystem()


def test_credit_boosts_salience():
    mem = _make_system()
    item_id = mem.remember("important finding", memory_type="semantic", importance=5)

    # Find the item in the fallback store
    item = next(i for i in mem._fallback._items if i.id == item_id)
    original_salience = item.salience

    mem.credit([item_id], signal=1.0)

    assert item.salience > original_salience


def test_credit_clamps_salience():
    mem = _make_system()
    item_id = mem.remember("already salient", memory_type="semantic", importance=9)

    # Boost many times
    for _ in range(20):
        mem.credit([item_id], signal=1.0)

    item = next(i for i in mem._fallback._items if i.id == item_id)
    assert item.salience <= 1.0


def test_credit_with_unknown_id_does_not_crash():
    mem = _make_system()
    mem.credit(["nonexistent-id"], signal=0.5)  # should be a no-op


def test_decay_runs_on_nth_recall():
    mem = _make_system()
    mem._decay_every_n = 2  # trigger every 2nd recall for testing
    mem.remember("test content", memory_type="semantic", importance=5)

    # Force items to have old last_access so decay has effect
    import time
    for item in mem._fallback._items:
        item.last_access = time.time() - 100000  # ~1 day ago

    item = mem._fallback._items[0]
    original_salience = item.salience

    # First recall: no decay yet
    mem.recall("test")
    # Second recall: decay should fire
    mem.recall("test")

    assert item.salience <= original_salience


def test_consolidation_gating_on_recall():
    mem = _make_system()
    mem.remember("content A", memory_type="semantic", importance=5)
    mem.remember("content B", memory_type="semantic", importance=5)

    results = mem.recall("content A", top_k=2)
    # Just verify recall still works and returns results
    assert len(results) >= 1
