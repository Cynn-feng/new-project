from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RAGContext:
    used: bool
    source_count: int
    context_text: str
    note: str


class RAGPipeline:
    def prepare_context(self, query: str) -> RAGContext:
        return RAGContext(
            used=False,
            source_count=0,
            context_text="",
            note=f"RAG pipeline is not connected to a corpus yet. Query retained: {query[:80]}",
        )
