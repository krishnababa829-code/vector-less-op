"""Token-aware HTML splitting at structural boundaries.

Splits large HTML documents at safe structural points (sections, articles)
when they exceed the token limit for LLM processing.
"""

import re
from dataclasses import dataclass
from typing import Callable

from bs4 import BeautifulSoup, Tag

from vnull.core.config import settings
from vnull.core.logging import get_logger

logger = get_logger(__name__)

_tokenizer = None


def get_tokenizer() -> Callable[[str], list]:
    """Get or create Qwen tokenizer for token counting."""
    global _tokenizer
    
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                trust_remote_code=True,
            )
            logger.info("Loaded Qwen tokenizer")
        except Exception as e:
            logger.warning(f"Failed to load Qwen tokenizer: {e}, using fallback")
            _tokenizer = lambda text: text.split()
    
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using Qwen tokenizer."""
    tokenizer = get_tokenizer()
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))
    return len(tokenizer(text))


@dataclass
class HTMLChunk:
    """A chunk of HTML with metadata."""
    content: str
    token_count: int
    chunk_index: int
    total_chunks: int
    split_tag: str | None = None


class HTMLSplitter:
    """Split HTML at structural boundaries when exceeding token limit.
    
    Splitting hierarchy (safest to least safe):
    1. <section> boundaries
    2. <article> boundaries
    3. <div> with semantic classes
    4. <h1>-<h6> headers
    5. <p> paragraphs (last resort)
    
    Example:
        >>> splitter = HTMLSplitter(max_tokens=6000)
        >>> chunks = splitter.split(large_html)
        >>> for chunk in chunks:
        ...     process_chunk(chunk.content)
    """
    
    SPLIT_PRIORITY = [
        "section",
        "article",
        "div",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p",
        "ul", "ol",
        "table",
    ]
    
    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_tokens: int = 100,
    ) -> None:
        """Initialize splitter.
        
        Args:
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Token overlap between chunks for context.
        """
        self.max_tokens = max_tokens or settings.max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
    
    def needs_splitting(self, html: str) -> bool:
        """Check if HTML exceeds token limit."""
        return count_tokens(html) > self.max_tokens
    
    def _find_split_points(self, soup: BeautifulSoup) -> list[tuple[int, str, Tag]]:
        """Find potential split points in document order.
        
        Returns list of (position, tag_name, element) tuples.
        """
        split_points: list[tuple[int, str, Tag]] = []
        html_str = str(soup)
        
        for tag_name in self.SPLIT_PRIORITY:
            for element in soup.find_all(tag_name):
                if not isinstance(element, Tag):
                    continue
                
                element_str = str(element)
                pos = html_str.find(element_str)
                if pos != -1:
                    split_points.append((pos, tag_name, element))
        
        split_points.sort(key=lambda x: x[0])
        return split_points
    
    def _split_at_midpoint(self, html: str, split_points: list[tuple[int, str, Tag]]) -> tuple[str, str, str]:
        """Find best split point near the middle.
        
        Returns:
            Tuple of (first_half, second_half, split_tag_name).
        """
        midpoint = len(html) // 2
        
        best_point = None
        best_distance = float("inf")
        best_tag = "unknown"
        
        for pos, tag_name, element in split_points:
            distance = abs(pos - midpoint)
            if distance < best_distance:
                best_distance = distance
                best_point = pos
                best_tag = tag_name
        
        if best_point is None:
            best_point = midpoint
            newline_pos = html.rfind("\n", 0, midpoint)
            if newline_pos > midpoint // 2:
                best_point = newline_pos
        
        return html[:best_point], html[best_point:], best_tag
    
    def split(self, html: str) -> list[HTMLChunk]:
        """Split HTML into token-limited chunks.
        
        Args:
            html: HTML content to split.
            
        Returns:
            List of HTMLChunk objects.
        """
        token_count = count_tokens(html)
        
        if token_count <= self.max_tokens:
            return [HTMLChunk(
                content=html,
                token_count=token_count,
                chunk_index=0,
                total_chunks=1,
            )]
        
        logger.info(
            "Splitting HTML",
            token_count=token_count,
            max_tokens=self.max_tokens,
        )
        
        chunks: list[HTMLChunk] = []
        remaining = html
        chunk_index = 0
        
        while remaining:
            remaining_tokens = count_tokens(remaining)
            
            if remaining_tokens <= self.max_tokens:
                chunks.append(HTMLChunk(
                    content=remaining,
                    token_count=remaining_tokens,
                    chunk_index=chunk_index,
                    total_chunks=0,
                ))
                break
            
            soup = BeautifulSoup(remaining, "lxml")
            split_points = self._find_split_points(soup)
            
            first_half, second_half, split_tag = self._split_at_midpoint(remaining, split_points)
            
            first_tokens = count_tokens(first_half)
            if first_tokens > self.max_tokens:
                sub_chunks = self.split(first_half)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                chunks.append(HTMLChunk(
                    content=first_half,
                    token_count=first_tokens,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    split_tag=split_tag,
                ))
                chunk_index += 1
            
            remaining = second_half
        
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        logger.info(
            "HTML split complete",
            total_chunks=total,
            avg_tokens=sum(c.token_count for c in chunks) // total if total > 0 else 0,
        )
        
        return chunks
    
    def split_with_overlap(self, html: str) -> list[HTMLChunk]:
        """Split HTML with overlapping context between chunks."""
        base_chunks = self.split(html)
        
        if len(base_chunks) <= 1:
            return base_chunks
        
        overlapped: list[HTMLChunk] = []
        
        for i, chunk in enumerate(base_chunks):
            content = chunk.content
            
            if i > 0:
                prev_content = base_chunks[i - 1].content
                overlap_chars = min(len(prev_content), self.overlap_tokens * 4)
                content = f"<!-- continued -->\n{prev_content[-overlap_chars:]}\n{content}"
            
            overlapped.append(HTMLChunk(
                content=content,
                token_count=count_tokens(content),
                chunk_index=i,
                total_chunks=len(base_chunks),
                split_tag=chunk.split_tag,
            ))
        
        return overlapped
