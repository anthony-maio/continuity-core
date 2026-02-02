from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


EmbedFn = Callable[[str], List[float]]
NliFn = Callable[[str, str], Dict[str, float]]


@dataclass
class StressResult:
    s_omega: float
    d_log: float
    d_sem: float
    v_top: float
    components: Dict[str, float] = field(default_factory=dict)
    contradictions: List[Tuple[str, str, float]] = field(default_factory=list)
    should_trigger: bool = False


class EpistemicStressMonitor:
    def __init__(self, nli_fn: NliFn | None = None, embed_fn: EmbedFn | None = None,
                 alpha: float = 0.4, beta: float = 0.35, gamma: float = 0.25,
                 trigger_threshold: float = 0.3) -> None:
        total = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.nli_fn = nli_fn
        self.embed_fn = embed_fn
        self.trigger_threshold = trigger_threshold

    def compute(self, statements: List[str], concept_contexts: Dict[str, List[str]] | None = None,
                graph_sparsity: float = 0.0) -> StressResult:
        d_log, contradictions = self._logical_dissonance(statements)
        d_sem = self._semantic_divergence(concept_contexts or {})
        v_top = graph_sparsity
        s_omega = (self.alpha * d_log) + (self.beta * d_sem) + (self.gamma * v_top)
        return StressResult(
            s_omega=s_omega,
            d_log=d_log,
            d_sem=d_sem,
            v_top=v_top,
            components={
                "logical": self.alpha * d_log,
                "semantic": self.beta * d_sem,
                "topological": self.gamma * v_top,
            },
            contradictions=contradictions,
            should_trigger=s_omega > self.trigger_threshold,
        )

    def _logical_dissonance(self, statements: List[str]) -> Tuple[float, List[Tuple[str, str, float]]]:
        if len(statements) < 2:
            return 0.0, []
        scores: List[float] = []
        contradictions: List[Tuple[str, str, float]] = []
        for i, s1 in enumerate(statements):
            for j, s2 in enumerate(statements[i + 1:], i + 1):
                score = self._contradiction_score(s1, s2)
                scores.append(score)
                if score > 0.5:
                    contradictions.append((s1, s2, score))
        d_log = sum(scores) / len(scores)
        return d_log, contradictions

    def _contradiction_score(self, s1: str, s2: str) -> float:
        if self.nli_fn:
            result = self.nli_fn(s1, s2)
            return float(result.get("contradiction", 0.0))
        return self._heuristic_contradiction(s1, s2)

    def _heuristic_contradiction(self, s1: str, s2: str) -> float:
        neg_words = {"not", "never", "no", "none", "cannot", "can't"}
        s1_tokens = set(s1.lower().split())
        s2_tokens = set(s2.lower().split())
        has_neg = bool(s1_tokens & neg_words) ^ bool(s2_tokens & neg_words)
        overlap = len(s1_tokens & s2_tokens)
        if has_neg and overlap >= 2:
            return 0.7
        return 0.0

    def _semantic_divergence(self, concept_contexts: Dict[str, List[str]]) -> float:
        if not concept_contexts or not self.embed_fn:
            return 0.0
        divergences: List[float] = []
        for concept, contexts in concept_contexts.items():
            if len(contexts) < 2:
                continue
            embeds = [self.embed_fn(f"{concept}: {ctx}") for ctx in contexts]
            for i in range(len(embeds)):
                for j in range(i + 1, len(embeds)):
                    divergences.append(1.0 - _cosine_similarity(embeds[i], embeds[j]))
        if not divergences:
            return 0.0
        return sum(divergences) / len(divergences)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)
