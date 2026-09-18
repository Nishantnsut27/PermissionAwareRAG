"""Token-aware recursive chunking with page provenance."""
from __future__ import annotations

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .models import Chunk, CleanedDocument


class TokenChunker:
    """Wraps a recursive splitter with an explicit token length function."""

    def __init__(
        self,
        encoding_name: str = config.TOKENIZER_ENCODING,
        chunk_size: int = config.CHUNK_SIZE_TOKENS,
        chunk_overlap: int = config.CHUNK_OVERLAP_TOKENS,
        separators: list[str] | None = None,
    ) -> None:
        self.encoding_name = encoding_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.count_tokens,
            separators=separators or config.TEXT_SEPARATORS,
            keep_separator=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return self.splitter.split_text(text)


def pages_covering(spans: list[tuple[int, int, int]],
                   start: int, end: int) -> list[int]:
    return sorted({number for number, span_start, span_end in spans
                   if span_start < end and span_end > start})


def chunk_document(
    base_metadata: dict,
    cleaned: CleanedDocument,
    chunker: TokenChunker,
) -> list[Chunk]:
    pieces = chunker.split(cleaned.text)
    chunks: list[Chunk] = []
    cursor = 0

    for index, piece in enumerate(pieces, start=1):
        start = cleaned.text.find(piece, cursor)
        if start == -1:
            start = cleaned.text.find(piece)
        if start == -1:
            start = cursor
        end = start + len(piece)
        cursor = start + 1

        page_numbers = pages_covering(cleaned.page_spans, start, end)
        chunks.append(Chunk(
            chunk_id=f"{base_metadata['document_id']}_chunk_{index:03d}",
            text=piece,
            document_id=base_metadata["document_id"],
            document_type=base_metadata["document_type"],
            seller_id=base_metadata["seller_id"],
            department=base_metadata.get("department"),
            classification=base_metadata["classification"],
            created_date=base_metadata.get("created_date"),
            owner=base_metadata.get("owner"),
            scenario_id=base_metadata.get("scenario_id"),
            scenario_ids=list(base_metadata.get("scenario_ids") or []),
            source_file=base_metadata["source_file"],
            source_format=base_metadata["source_format"],
            page_numbers=page_numbers,
            chunk_index=index,
            token_count=chunker.count_tokens(piece),
            char_count=len(piece),
        ))
    return chunks
