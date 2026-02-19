"""Tests for MRA cache on TieredMemorySystem."""

import time
from continuity_core.memory.system import MRACache
from continuity_core.mra.stress import StressResult
from continuity_core.mra.voids import VoidReport


def _make_stress(s_omega=0.5, should_trigger=True):
    return StressResult(
        s_omega=s_omega, d_log=0.3, d_sem=0.1, v_top=0.1,
        should_trigger=should_trigger,
    )


def test_cache_starts_stale():
    cache = MRACache()
    assert cache.is_stale()


def test_cache_round_trip():
    cache = MRACache(staleness_sec=300.0)
    stress = _make_stress()
    voids = VoidReport(void_pairs=[], questions=["What connects A and B?"])

    cache.last_stress = stress
    cache.last_voids = voids
    cache.updated_at = time.time()

    assert not cache.is_stale()
    assert cache.last_stress.s_omega == 0.5
    assert len(cache.last_voids.questions) == 1


def test_cache_expires():
    cache = MRACache(staleness_sec=1.0)
    cache.last_stress = _make_stress()
    cache.updated_at = time.time() - 2.0  # 2 seconds ago, stale > 1s

    assert cache.is_stale()


def test_cache_not_stale_within_window():
    now = time.time()
    cache = MRACache(staleness_sec=300.0)
    cache.last_stress = _make_stress()
    cache.updated_at = now

    assert not cache.is_stale(now=now + 100)  # 100s < 300s window
    assert cache.is_stale(now=now + 400)      # 400s > 300s window
