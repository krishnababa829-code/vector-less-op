"""Iterative Explorer with MCTS-lite for multi-path retrieval.

Layer 3 of the retrieval pipeline. Explores chunks, extracts facts,
and synthesizes answers. Supports parent exploration.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from vnull.core.llm_client import LLMClient
from vnull.core.logging import get_logger
from vnull.indexer.toc_builder import TableOfContents, ToCEntry

logger = get_logger(__name__)

EXTRACT_SYSTEM = """You are a fact extraction system. Extract relevant facts from the given text that answer the query.

Rules:
1. Extract ONLY facts relevant to the query
2. Be concise but complete
3. If the text doesn't contain relevant information, say "NO_RELEVANT_FACTS"
4. If you need more context, output: {"action": "explore_parent", "target": "<parent_chunk_id>"}
5. Otherwise, output the extracted facts as plain text"""

SYNTHESIS_SYSTEM = """You are an answer synthesis system. Combine extracted facts into a coherent answer.

Rules:
1. Synthesize facts into a clear, complete answer
2. Resolve any contradictions by preferring more specific information
3. If facts are insufficient, acknowledge limitations
4. Be concise but thorough
5. Do not add information not present in the facts"""


@dataclass
class ExtractionResult:
    """Result of fact extraction from a chunk."""
    chunk_id: str
    facts: str
    needs_parent: bool = False
    parent_target: str | None = None


@dataclass
class ExplorationResult:
    """Result of iterative exploration."""
    query: str
    extractions: list[ExtractionResult]
    synthesis: str
    chunks_explored: int
    parent_explorations: int = 0


class IterativeExplorer:
    """Explore chunks and synthesize answers.
    
    Features:
    - Multi-path exploration of multiple chunks
    - MCTS-lite: Can request parent context
    - KV cache flushing between extractions
    - Final synthesis of all facts
    
    Example:
        >>> explorer = IterativeExplorer(toc)
        >>> result = await explorer.explore("How to auth?", ["chunk1", "chunk2"])
        >>> print(result.synthesis)
    """
    
    EXPLORE_PATTERN = re.compile(r'\{\s*"action"\s*:\s*"explore_parent"')
    
    def __init__(
        self,
        toc: TableOfContents,
        llm_client: LLMClient | None = None,
        max_parent_depth: int = 2,
    ) -> None:
        self.toc = toc
        self.llm = llm_client or LLMClient()
        self.max_parent_depth = max_parent_depth
    
    async def _extract_facts(
        self,
        query: str,
        entry: ToCEntry,
    ) -> ExtractionResult:
        """Extract facts from a single chunk."""
        prompt = f"""Query: {query}

Chunk ID: {entry.chunk_id}
Header: {entry.header}
Parent ID: {entry.parent_id or 'None'}

Content:
{entry.raw_markdown[:3000]}

Extract relevant facts:"""
        
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=EXTRACT_SYSTEM,
            temperature=0.1,
        )
        
        content = response.content.strip()
        
        if self.EXPLORE_PATTERN.search(content):
            try:
                data = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
                return ExtractionResult(
                    chunk_id=entry.chunk_id,
                    facts="",
                    needs_parent=True,
                    parent_target=data.get("target", entry.parent_id),
                )
            except (json.JSONDecodeError, AttributeError):
                pass
        
        if "NO_RELEVANT_FACTS" in content:
            return ExtractionResult(chunk_id=entry.chunk_id, facts="")
        
        return ExtractionResult(chunk_id=entry.chunk_id, facts=content)
    
    async def _synthesize(self, query: str, extractions: list[ExtractionResult]) -> str:
        """Synthesize facts into final answer."""
        facts_text = "\n\n".join([
            f"From {e.chunk_id}:\n{e.facts}"
            for e in extractions if e.facts
        ])
        
        if not facts_text.strip():
            return "I could not find relevant information to answer this query."
        
        prompt = f"""Query: {query}

Extracted Facts:
{facts_text}

Synthesize a complete answer:"""
        
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=SYNTHESIS_SYSTEM,
            temperature=0.3,
        )
        
        return response.content.strip()
    
    async def explore(
        self,
        query: str,
        chunk_ids: list[str],
    ) -> ExplorationResult:
        """Explore chunks and synthesize answer."""
        extractions: list[ExtractionResult] = []
        parent_explorations = 0
        explored_ids = set()
        
        to_explore = list(chunk_ids)
        
        while to_explore:
            chunk_id = to_explore.pop(0)
            
            if chunk_id in explored_ids:
                continue
            explored_ids.add(chunk_id)
            
            entry = self.toc.get_entry(chunk_id)
            if not entry:
                continue
            
            extraction = await self._extract_facts(query, entry)
            extractions.append(extraction)
            
            if extraction.needs_parent and parent_explorations < self.max_parent_depth:
                parent_id = extraction.parent_target or entry.parent_id
                if parent_id and parent_id not in explored_ids:
                    to_explore.insert(0, parent_id)
                    parent_explorations += 1
            
            if len(explored_ids) < len(chunk_ids) + self.max_parent_depth:
                await self.llm.flush_kv_cache()
        
        synthesis = await self._synthesize(query, extractions)
        
        logger.info(
            "Exploration complete",
            query=query[:50],
            chunks=len(explored_ids),
            parents=parent_explorations,
        )
        
        return ExplorationResult(
            query=query,
            extractions=extractions,
            synthesis=synthesis,
            chunks_explored=len(explored_ids),
            parent_explorations=parent_explorations,
        )
