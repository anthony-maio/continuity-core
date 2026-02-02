from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from continuity_core.config import C2Config, load_config
from continuity_core.event_log import EventLog
from continuity_core.memory.embeddings import build_embedder
from continuity_core.memory.stores import InMemoryStore
from continuity_core.storage import Neo4jGraphStore, PostgresEventStore, QdrantMemoryStore, QdrantResult, RedisWorkingContext


@dataclass
class ScoredMemory:
    id: str
    score: float
    content: str
    memory_type: str
    payload: Dict[str, Any]


class TieredMemorySystem:
    def __init__(self, config: Optional[C2Config] = None) -> None:
        self.config = config or load_config()
        self.embedder = build_embedder(self.config)
        self._event_log = self._init_event_log()
        self._redis = self._init_redis()
        self._qdrant, self._fallback = self._init_qdrant()
        self._neo4j = self._init_neo4j()

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    @property
    def redis(self) -> Optional[RedisWorkingContext]:
        return self._redis

    @property
    def qdrant(self) -> Optional[QdrantMemoryStore]:
        return self._qdrant

    @property
    def neo4j(self) -> Optional[Neo4jGraphStore]:
        return self._neo4j

    def write_event(self, actor: str, intent: str, inp: str, out: str,
                    tags: Optional[List[str]] = None, metadata: Optional[Dict[str, str]] = None) -> None:
        self._event_log.log(actor, intent, inp, out, tags=tags, metadata=metadata)

    def append_working_context(self, thread_id: str, message: Dict[str, Any]) -> None:
        if self._redis is None:
            return
        self._redis.append(thread_id, message)

    def get_working_context(self, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if self._redis is None:
            return []
        return self._redis.get_recent(thread_id, limit=limit)

    def remember(self, content: str, memory_type: str, importance: int = 5,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        if self._qdrant is not None:
            return self._qdrant.remember(content, memory_type, importance, metadata)
        # Fallback store for offline tests
        if self._fallback is None:
            raise RuntimeError("No memory store available")
        item = self._fallback.add(content, salience=float(importance) / 10.0, metadata=metadata)
        return item.id

    def recall(self, query: str, top_k: int = 5, type_filter: Optional[str] = None) -> List[ScoredMemory]:
        if self._qdrant is not None:
            results = self._qdrant.recall(query, top_k=top_k, type_filter=type_filter)
            return self._score_results(results)
        if self._fallback is None:
            return []
        scored = self._fallback.query(query, top_k=top_k)
        out: List[ScoredMemory] = []
        for item, sim in scored:
            out.append(ScoredMemory(
                id=item.id,
                score=sim,
                content=str(item.content),
                memory_type=item.metadata.get("type", "unknown"),
                payload=item.metadata,
            ))
        return out

    def _score_results(self, results: List[QdrantResult]) -> List[ScoredMemory]:
        out: List[ScoredMemory] = []
        for r in results:
            payload = r.payload
            recency = self._recency_score(payload.get("last_accessed"))
            importance = float(payload.get("importance", 5)) / 10.0
            score = (0.5 * r.score) + (0.3 * recency) + (0.2 * importance)
            out.append(ScoredMemory(
                id=r.id,
                score=score,
                content=payload.get("content", ""),
                memory_type=payload.get("type", "unknown"),
                payload=payload,
            ))
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _recency_score(self, last_accessed: Optional[float]) -> float:
        if last_accessed is None:
            return 0.0
        age_days = (time.time() - float(last_accessed)) / 86400.0
        return math.exp(-age_days * math.log(2.0) / max(0.1, self.config.recency_half_life_days))

    def _init_event_log(self) -> EventLog:
        try:
            store = PostgresEventStore(self.config.postgres_url)
            return EventLog(store)
        except Exception:
            return EventLog()

    def _init_redis(self) -> Optional[RedisWorkingContext]:
        try:
            return RedisWorkingContext(self.config.redis_url)
        except Exception:
            return None

    def _init_qdrant(self) -> tuple[Optional[QdrantMemoryStore], Optional[InMemoryStore]]:
        try:
            vector = self.embedder.embed("seed")
            store = QdrantMemoryStore(
                url=self.config.qdrant_url,
                collection="c2_memories",
                embed_fn=self.embedder.embed,
                vector_size=len(vector),
            )
            return store, None
        except Exception:
            fallback = InMemoryStore(capacity=5000, embed_fn=self.embedder.embed)
            return None, fallback

    def _init_neo4j(self) -> Optional[Neo4jGraphStore]:
        try:
            return Neo4jGraphStore(self.config.neo4j_uri, self.config.neo4j_user, self.config.neo4j_password)
        except Exception:
            return None
