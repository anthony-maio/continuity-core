from continuity_core.event_log import EventLog
from continuity_core.ingest.chunker import chunk_text
from continuity_core.ingest.pipeline import IngestPipeline


class StubMemorySystem:
    def __init__(self) -> None:
        self.event_log = EventLog()
        self.neo4j = None
        self.remembered = []

    def remember(self, content: str, memory_type: str, importance: int = 5, metadata=None) -> str:
        self.remembered.append((content, memory_type, importance, metadata))
        return "mem-id"


def test_chunk_text_overlap():
    text = "a" * 120
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 3
    assert chunks[0][-10:] == chunks[1][:10]


def test_ingest_pipeline_counts(tmp_path):
    sample = tmp_path / "note.txt"
    sample.write_text("hello world " * 30, encoding="utf-8")
    memory = StubMemorySystem()
    pipeline = IngestPipeline(
        memory_system=memory,
        allowed_exts={".txt"},
        chunk_size=60,
        chunk_overlap=0,
        max_bytes=100000,
    )
    result = pipeline.ingest_paths([str(tmp_path)])
    assert result.files_seen == 1
    assert result.docs_ingested == 1
    assert result.chunks_ingested >= 1
    assert len(memory.remembered) == result.chunks_ingested
    assert memory.event_log.tail(1)
