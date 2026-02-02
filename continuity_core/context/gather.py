from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from continuity_core.context.composer import Candidate
from continuity_core.event_log import EventLog
from continuity_core.memory.system import ScoredMemory, TieredMemorySystem
from continuity_core.storage.neo4j import Neo4jGraphStore


class CandidateGatherer:
    def __init__(self, memory_system: TieredMemorySystem) -> None:
        self._memory = memory_system

    def gather(self, query: str, thread_id: Optional[str] = None, top_k: int = 8) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
        candidates: List[Candidate] = []
        working_context: List[Dict[str, Any]] = []
        now = time.time()

        if thread_id:
            working_context = self._memory.get_working_context(thread_id, limit=12)
            for msg in working_context:
                text = str(msg.get("content", ""))
                candidates.append(_candidate_from_text("working", text, now, base_relevance=0.9))

        memories = self._memory.recall(query, top_k=top_k)
        for mem in memories:
            candidates.append(_candidate_from_memory(mem, now))

        if self._memory.neo4j is not None:
            candidates.extend(self._graph_candidates(query, self._memory.neo4j, now))

        events = self._memory.event_log.query(limit=8)
        for event in events:
            text = f"{event.intent}: {event.output}"
            candidates.append(_candidate_from_text("event", text, now, base_relevance=0.4))

        return candidates, working_context

    def _graph_candidates(self, query: str, graph: Neo4jGraphStore, now: float) -> List[Candidate]:
        nodes = graph.query_nodes(text=query, node_types=None, limit=8)
        out: List[Candidate] = []
        for node in nodes:
            name = node.get("name", "")
            desc = node.get("description", "") or ""
            text = name if not desc else f"{name} - {desc}"
            out.append(_candidate_from_text("graph", text, now, base_relevance=0.6))
        return out


def _candidate_from_memory(mem: ScoredMemory, now: float) -> Candidate:
    text = mem.content
    recency = now - float(mem.payload.get("last_accessed", now))
    return Candidate(
        id=mem.id,
        text=text,
        store=mem.memory_type,
        token_cost=_token_cost(text),
        relevance=mem.score,
        recency_sec=recency,
        centrality=0.4,
        confidence=float(mem.payload.get("confidence", 0.8)),
        task_match=0.8,
        salience=float(mem.payload.get("importance", 5)) / 10.0,
    )


def _candidate_from_text(store: str, text: str, now: float, base_relevance: float = 0.5) -> Candidate:
    return Candidate(
        id=f"{store}:{hash(text)}",
        text=text,
        store=store,
        token_cost=_token_cost(text),
        relevance=base_relevance,
        recency_sec=0.0,
        centrality=0.3,
        confidence=0.7,
        task_match=0.7,
        salience=0.6,
    )


def _token_cost(text: str) -> int:
    return max(1, int(len(text) * 0.25))
