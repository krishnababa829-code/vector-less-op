"""Semantic Markdown chunking at header boundaries.

Chunks Markdown strictly at header boundaries (#, ##, etc.)
without severing paragraphs. Extracts bookend metadata.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from vnull.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarkdownChunk:
    """A chunk of Markdown with bookend metadata."""
    chunk_id: str
    content: str
    header: str
    header_level: int
    first_sentence: str
    last_sentence: str
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    source_url: Optional[str] = None
    char_count: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        self.char_count = len(self.content)
        self.word_count = len(self.content.split())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "header": self.header,
            "header_level": self.header_level,
            "first_sentence": self.first_sentence,
            "last_sentence": self.last_sentence,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "source_url": self.source_url,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "content": self.content,
        }


class MarkdownChunker:
    """Chunk Markdown at header boundaries with bookend extraction.
    
    Features:
    - Splits at # headers while preserving hierarchy
    - Extracts first and last sentences as bookend metadata
    - Maintains parent-child relationships
    - Generates stable chunk IDs
    
    Example:
        >>> chunker = MarkdownChunker()
        >>> chunks = chunker.chunk(markdown_content)
        >>> for chunk in chunks:
        ...     print(f"{chunk.header}: {chunk.first_sentence[:50]}...")
    """
    
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    SENTENCE_PATTERN = re.compile(r"[^.!?]*[.!?]", re.DOTALL)
    
    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 10000,
        include_header_in_content: bool = True,
    ) -> None:
        """Initialize chunker.
        
        Args:
            min_chunk_size: Minimum characters per chunk.
            max_chunk_size: Maximum characters per chunk.
            include_header_in_content: Include header in chunk content.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.include_header_in_content = include_header_in_content
    
    def _generate_chunk_id(self, header: str, content: str) -> str:
        """Generate stable chunk ID from header and content."""
        combined = f"{header}:{content[:100]}"
        return hashlib.sha256(combined.encode()).hexdigest()[:12]
    
    def _extract_first_sentence(self, text: str) -> str:
        """Extract first sentence from text."""
        text = re.sub(r"^#+\s+.+\n", "", text).strip()
        text = re.sub(r"\s+", " ", text)
        
        match = self.SENTENCE_PATTERN.search(text)
        if match:
            return match.group(0).strip()
        
        words = text.split()[:30]
        return " ".join(words) + ("..." if len(words) == 30 else "")
    
    def _extract_last_sentence(self, text: str) -> str:
        """Extract last sentence from text."""
        text = re.sub(r"\s+", " ", text).strip()
        
        sentences = self.SENTENCE_PATTERN.findall(text)
        if sentences:
            return sentences[-1].strip()
        
        words = text.split()[-30:]
        return ("..." if len(text.split()) > 30 else "") + " ".join(words)
    
    def _find_headers(self, markdown: str) -> list[tuple[int, int, str, int]]:
        """Find all headers with positions.
        
        Returns:
            List of (start_pos, end_pos, header_text, level) tuples.
        """
        headers = []
        for match in self.HEADER_PATTERN.finditer(markdown):
            level = len(match.group(1))
            header_text = match.group(2).strip()
            headers.append((match.start(), match.end(), header_text, level))
        return headers
    
    def chunk(
        self,
        markdown: str,
        source_url: str | None = None,
    ) -> list[MarkdownChunk]:
        """Chunk Markdown at header boundaries.
        
        Args:
            markdown: Markdown content to chunk.
            source_url: Optional source URL for metadata.
            
        Returns:
            List of MarkdownChunk objects.
        """
        headers = self._find_headers(markdown)
        
        if not headers:
            chunk_id = self._generate_chunk_id("root", markdown)
            return [MarkdownChunk(
                chunk_id=chunk_id,
                content=markdown,
                header="Document",
                header_level=0,
                first_sentence=self._extract_first_sentence(markdown),
                last_sentence=self._extract_last_sentence(markdown),
                source_url=source_url,
            )]
        
        chunks: list[MarkdownChunk] = []
        header_stack: list[tuple[str, int]] = []
        
        if headers[0][0] > 0:
            preamble = markdown[:headers[0][0]].strip()
            if len(preamble) >= self.min_chunk_size:
                chunk_id = self._generate_chunk_id("preamble", preamble)
                chunks.append(MarkdownChunk(
                    chunk_id=chunk_id,
                    content=preamble,
                    header="Preamble",
                    header_level=0,
                    first_sentence=self._extract_first_sentence(preamble),
                    last_sentence=self._extract_last_sentence(preamble),
                    source_url=source_url,
                ))
        
        for i, (start, end, header_text, level) in enumerate(headers):
            if i + 1 < len(headers):
                content_end = headers[i + 1][0]
            else:
                content_end = len(markdown)
            
            if self.include_header_in_content:
                content = markdown[start:content_end].strip()
            else:
                content = markdown[end:content_end].strip()
            
            if len(content) < self.min_chunk_size:
                continue
            
            chunk_id = self._generate_chunk_id(header_text, content)
            
            while header_stack and header_stack[-1][1] >= level:
                header_stack.pop()
            
            parent_id = header_stack[-1][0] if header_stack else None
            
            chunk = MarkdownChunk(
                chunk_id=chunk_id,
                content=content,
                header=header_text,
                header_level=level,
                first_sentence=self._extract_first_sentence(content),
                last_sentence=self._extract_last_sentence(content),
                parent_id=parent_id,
                source_url=source_url,
            )
            chunks.append(chunk)
            
            header_stack.append((chunk_id, level))
        
        chunk_map = {c.chunk_id: c for c in chunks}
        for chunk in chunks:
            if chunk.parent_id and chunk.parent_id in chunk_map:
                chunk_map[chunk.parent_id].children_ids.append(chunk.chunk_id)
        
        logger.info(
            "Markdown chunked",
            total_chunks=len(chunks),
            source_url=source_url,
        )
        
        return chunks
    
    def chunk_file(self, filepath: str, source_url: str | None = None) -> list[MarkdownChunk]:
        """Chunk a Markdown file."""
        from pathlib import Path
        content = Path(filepath).read_text(encoding="utf-8")
        
        url_match = re.search(r"<!-- Source: (.+?) -->", content)
        if url_match and not source_url:
            source_url = url_match.group(1)
        
        return self.chunk(content, source_url=source_url)
