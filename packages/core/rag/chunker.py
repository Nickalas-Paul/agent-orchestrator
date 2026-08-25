"""Character-based document chunking strategies."""

from __future__ import annotations

from packages.core.logging.logger import get_logger
from packages.core.rag.models import ChunkMetadata, DocumentChunk

logger = get_logger("rag.chunker")

_VALID_STRATEGIES = {"fixed", "recursive", "semantic"}


class DocumentChunker:
    """Split document text into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "recursive",
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Character overlap between consecutive fixed slices.
            strategy: ``fixed``, ``recursive``, or ``semantic``.
        """
        self._chunk_size = max(1, chunk_size)
        self._chunk_overlap = max(0, chunk_overlap)
        self._strategy = strategy if strategy in _VALID_STRATEGIES else "recursive"

    def chunk_document(self, text: str, metadata: ChunkMetadata) -> list[DocumentChunk]:
        """Split ``text`` and attach per-chunk metadata copies."""
        if not text:
            return []

        if self._strategy == "fixed":
            pieces = self._fixed_chunk(text)
        elif self._strategy == "semantic":
            pieces = self._semantic_chunk(text)
        else:
            pieces = self._recursive_chunk(text)

        total = len(pieces)
        chunks: list[DocumentChunk] = []
        for index, piece in enumerate(pieces):
            chunk_meta = metadata.model_copy(
                update={"chunk_index": index, "total_chunks": total}
            )
            chunks.append(DocumentChunk(text=piece, metadata=chunk_meta))
        return chunks

    def _fixed_chunk(self, text: str) -> list[str]:
        """Split every ``chunk_size`` characters with ``chunk_overlap`` overlap."""
        if not text:
            return []
        size = self._chunk_size
        step = max(1, size - self._chunk_overlap)
        pieces: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + size, length)
            pieces.append(text[start:end])
            if end >= length:
                break
            start += step
        return pieces

    def _recursive_chunk(self, text: str) -> list[str]:
        """Split on paragraphs, then lines, then sentences, then fixed size."""
        return self._recursive_split(text, ["\n\n", "\n", ". "])

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Split ``text`` with the remaining separator hierarchy."""
        if not text:
            return []
        if len(text) <= self._chunk_size:
            return [text]
        if not separators:
            return self._fixed_chunk(text)

        separator = separators[0]
        remaining = separators[1:]
        if separator not in text:
            return self._recursive_split(text, remaining)

        raw_parts = text.split(separator)
        pieces: list[str] = []
        for index, part in enumerate(raw_parts):
            if index < len(raw_parts) - 1:
                piece = part + separator
            else:
                piece = part
            if not piece:
                continue
            if len(piece) <= self._chunk_size:
                pieces.append(piece)
            else:
                pieces.extend(self._recursive_split(piece, remaining))
        return pieces or self._fixed_chunk(text)

    def _semantic_chunk(self, text: str) -> list[str]:
        """Stub: semantic chunking is deferred; fall back to recursive splitting."""
        logger.warning(
            "semantic_chunking_not_implemented",
            message=(
                "Semantic chunking requires an embedding model and is not yet "
                "implemented; falling back to recursive chunking."
            ),
        )
        return self._recursive_chunk(text)
