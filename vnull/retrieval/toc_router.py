"""ToC Router - LLM-based chunk selection from Table of Contents.

Layer 2 of the retrieval pipeline. Routes queries to relevant
chunks by having the LLM evaluate signposts.
"""

import json
import re
from dataclasses import dataclass

from vnull.core.llm_client import LLMClient
from vnull.core.logging import get_logger
from vnull.indexer.toc_builder import TableOfContents

logger = get_logger(__name__)

ROUTER_SYSTEM = """You are a document routing system. Given a query and a table of contents with signposts, select the most relevant chunks.

Each entry has:
- chunk_id: Unique identifier
- signpost: [Core Theme] + [Key Entities] + [Questions Answered]
- header: Section header

Your task:
1. Analyze the query
2. Match against signposts
3. Return relevant chunk_ids as JSON array

Rules:
- Return 1-5 most relevant chunk_ids
- If no chunks are relevant, return empty array []
- Order by relevance (most relevant first)
- Output ONLY the JSON array, nothing else

Example output: ["abc123", "def456", "ghi789"]"""


@dataclass
class RouteResult:
    """Result of ToC routing."""
    query: str
    chunk_ids: list[str]
    confidence: float
    
    @property
    def has_matches(self) -> bool:
        return len(self.chunk_ids) > 0


class ToCRouter:
    """Route queries to relevant chunks via LLM.
    
    Example:
        >>> router = ToCRouter(toc)
        >>> result = await router.route("How to authenticate?")
        >>> print(result.chunk_ids)
        ['auth_chunk_1', 'auth_chunk_2']
    """
    
    def __init__(
        self,
        toc: TableOfContents,
        llm_client: LLMClient | None = None,
        max_signposts_per_request: int = 50,
    ) -> None:
        self.toc = toc
        self.llm = llm_client or LLMClient()
        self.max_signposts = max_signposts_per_request
    
    def _format_signposts(self, start: int = 0, count: int | None = None) -> str:
        """Format signposts for LLM prompt."""
        entries = self.toc.entries[start:start + (count or self.max_signposts)]
        lines = []
        for e in entries:
            lines.append(f"- {e.chunk_id}: {e.dense_signpost} (Header: {e.header})")
        return "\n".join(lines)
    
    async def route(self, query: str) -> RouteResult:
        """Route query to relevant chunks."""
        signposts_text = self._format_signposts()
        
        prompt = f"""Query: {query}

Table of Contents:
{signposts_text}

Return the chunk_ids of relevant sections as a JSON array:"""
        
        try:
            result = await self.llm.complete_json(
                prompt=prompt,
                system_prompt=ROUTER_SYSTEM,
                temperature=0.1,
            )
            
            if isinstance(result, list):
                chunk_ids = [str(cid) for cid in result if isinstance(cid, str)]
            else:
                chunk_ids = []
            
            valid_ids = [cid for cid in chunk_ids if self.toc.get_entry(cid)]
            
            confidence = len(valid_ids) / max(len(chunk_ids), 1) if chunk_ids else 0.0
            
            logger.debug("Route complete", query=query[:50], matches=len(valid_ids))
            
            return RouteResult(query=query, chunk_ids=valid_ids, confidence=confidence)
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return RouteResult(query=query, chunk_ids=[], confidence=0.0)
    
    async def route_multi(self, queries: list[str]) -> list[RouteResult]:
        """Route multiple queries and merge results."""
        results = []
        seen_ids = set()
        
        for query in queries:
            result = await self.route(query)
            for cid in result.chunk_ids:
                if cid not in seen_ids:
                    seen_ids.add(cid)
            results.append(result)
        
        return results
