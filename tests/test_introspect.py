"""Tests for the introspect tool wiring and graph sparsity."""

from continuity_core.mcp.tools.introspect import _graph_sparsity


def test_graph_sparsity_empty():
    assert _graph_sparsity({}) == 1.0


def test_graph_sparsity_single_node():
    assert _graph_sparsity({"a": set()}) == 1.0


def test_graph_sparsity_complete_graph():
    # 3-node complete directed graph: 6 edges out of 6 possible
    graph = {
        "a": {"b", "c"},
        "b": {"a", "c"},
        "c": {"a", "b"},
    }
    sparsity = _graph_sparsity(graph)
    assert sparsity < 0.01  # essentially 0 — fully connected


def test_graph_sparsity_sparse_graph():
    # 4 nodes, only 2 edges out of 12 possible
    graph = {
        "a": {"b"},
        "b": set(),
        "c": {"d"},
        "d": set(),
    }
    sparsity = _graph_sparsity(graph)
    expected = 1.0 - (2.0 / 12.0)
    assert abs(sparsity - expected) < 0.01


def test_graph_sparsity_linear_chain():
    # a -> b -> c -> d: 3 edges, 4 nodes, 12 possible
    graph = {
        "a": {"b"},
        "b": {"c"},
        "c": {"d"},
        "d": set(),
    }
    sparsity = _graph_sparsity(graph)
    assert 0.7 < sparsity < 0.8  # 1 - 3/12 = 0.75
