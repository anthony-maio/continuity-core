from continuity_core.context.pipeline import ContextPipeline
from continuity_core.event_log import EventLog
from continuity_core.memory.system import ScoredMemory


class StubMemorySystem:
    def __init__(self) -> None:
        self.event_log = EventLog()
        self.neo4j = None

    def get_working_context(self, thread_id: str, limit: int = 20):
        return [{"role": "user", "content": "hello"}]

    def recall(self, query: str, top_k: int = 5, type_filter=None):
        return [
            ScoredMemory(id="mem1", score=0.9, content="fact one", memory_type="semantic", payload={"importance": 9}),
            ScoredMemory(id="mem2", score=0.6, content="fact two", memory_type="episodic", payload={"importance": 5}),
        ]


def test_context_pipeline_builds_prompt_pack():
    pipeline = ContextPipeline(memory_system=StubMemorySystem())
    result = pipeline.run(query="test", thread_id="t1")
    assert result.chosen
    assert result.working_context
