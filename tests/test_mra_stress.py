"""Tests for EpistemicStressMonitor with deep tension detection."""

import math
from continuity_core.mra.stress import EpistemicStressMonitor, _cosine_similarity


def _simple_embed(text: str):
    """Deterministic fake embedder: bag-of-chars normalized."""
    vec = [0.0] * 26
    for c in text.lower():
        if 'a' <= c <= 'z':
            vec[ord(c) - ord('a')] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def test_basic_no_contradiction():
    monitor = EpistemicStressMonitor()
    result = monitor.compute(["The sky is blue", "Water is wet"])
    assert result.d_log == 0.0
    assert result.contradictions == []
    assert result.deep_tensions == []


def test_heuristic_contradiction():
    monitor = EpistemicStressMonitor()
    result = monitor.compute([
        "The system is fast and reliable",
        "The system is not fast or reliable",
    ])
    assert result.d_log > 0.5
    assert len(result.contradictions) == 1


def test_deep_tension_with_embedder():
    """High contradiction + high similarity = deep tension."""
    def mock_nli(s1, s2):
        return {"contradiction": 0.9}

    monitor = EpistemicStressMonitor(
        nli_fn=mock_nli,
        embed_fn=_simple_embed,
        deep_tension_contradiction_threshold=0.7,
        deep_tension_similarity_threshold=0.3,
    )
    # Very similar sentences that "contradict" per NLI:
    result = monitor.compute([
        "Neural networks generalize well",
        "Neural networks generalize poorly",
    ])
    assert len(result.deep_tensions) >= 1
    # The amplified score should be higher than the raw 0.9
    dt_score = result.deep_tensions[0][2]
    assert dt_score > 0.9


def test_deep_tension_not_triggered_when_dissimilar():
    """Contradictory but dissimilar statements shouldn't trigger deep tension."""
    def mock_nli(s1, s2):
        return {"contradiction": 0.8}

    def _ortho_embed(text):
        # Return orthogonal embeddings so similarity is ~0.
        if "cats" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    monitor = EpistemicStressMonitor(
        nli_fn=mock_nli,
        embed_fn=_ortho_embed,
        deep_tension_similarity_threshold=0.6,
    )
    result = monitor.compute(["cats are great", "dogs are terrible"])
    assert len(result.deep_tensions) == 0


def test_semantic_divergence_with_embedder():
    monitor = EpistemicStressMonitor(embed_fn=_simple_embed)
    result = monitor.compute(
        statements=[],
        concept_contexts={"AI": ["AI is transformative", "AI is a buzzword"]},
    )
    assert result.d_sem > 0.0


def test_semantic_divergence_without_embedder():
    monitor = EpistemicStressMonitor(embed_fn=None)
    result = monitor.compute(
        statements=[],
        concept_contexts={"AI": ["AI is transformative", "AI is a buzzword"]},
    )
    assert result.d_sem == 0.0


def test_single_statement():
    monitor = EpistemicStressMonitor()
    result = monitor.compute(["only one statement"])
    assert result.d_log == 0.0
    assert result.s_omega == 0.0


def test_graph_sparsity_contributes():
    monitor = EpistemicStressMonitor()
    result = monitor.compute([], graph_sparsity=0.8)
    assert result.v_top == 0.8
    assert result.s_omega > 0.0


def test_trigger_threshold():
    monitor = EpistemicStressMonitor(trigger_threshold=0.1)
    result = monitor.compute([], graph_sparsity=0.8)
    assert result.should_trigger is True

    monitor2 = EpistemicStressMonitor(trigger_threshold=0.99)
    result2 = monitor2.compute([], graph_sparsity=0.1)
    assert result2.should_trigger is False
