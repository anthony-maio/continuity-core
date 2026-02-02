from __future__ import annotations

from typing import Any, Dict

from continuity_core.mra import EpistemicStressMonitor, VoidDetector


def _normalize_graph(raw: Dict[str, Any]) -> Dict[str, set]:
    graph: Dict[str, set] = {}
    for node, neighbors in raw.items():
        graph[str(node)] = {str(n) for n in (neighbors or [])}
    return graph


def introspect(arguments: Dict[str, Any]) -> Dict[str, Any]:
    statements = arguments.get("statements") or []
    concept_contexts = arguments.get("concept_contexts") or {}
    graph_raw = arguments.get("graph") or {}

    monitor = EpistemicStressMonitor()
    stress = monitor.compute(statements, concept_contexts=concept_contexts, graph_sparsity=0.0)

    void_detector = VoidDetector()
    graph = _normalize_graph(graph_raw)
    voids = void_detector.detect_voids(graph) if graph else None

    return {
        "stress": {
            "s_omega": stress.s_omega,
            "d_log": stress.d_log,
            "d_sem": stress.d_sem,
            "v_top": stress.v_top,
            "components": stress.components,
            "should_trigger": stress.should_trigger,
            "contradictions": stress.contradictions,
        },
        "voids": {
            "pairs": [
                {"a": list(a), "b": list(b), "edges_between": edges}
                for a, b, edges in (voids.void_pairs if voids else [])
            ],
            "questions": (voids.questions if voids else []),
        },
    }
