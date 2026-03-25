"""BM25 Fallback - Lexical search when LLM routing fails.

Layer 4 (Ultimate Fallback) of the retrieval pipeline.
Uses rank_bm25 for lexical matching on bookends and content.
"""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from vnull.core.logging import get_logger
from vnull.indexer.toc_builder import TableOfContents, ToCEntry

logger = get_logger(__name__)


@dataclass
class BM25Result:
    """Result of BM25 search."""
    query: str
    matches: list[tuple[ToCEntry, float]]
    
    @property
    def top_entry(self) -> ToCEntry | None:
        return self.matches[0][0] if self.matches else None
    
    @property
    def top_score(self) -> float:
        return self.matches[0][1] if self.matches else 0.0


class BM25Fallback:
    """BM25 lexical search fallback.
    
    Searches against:
    - first_sentence (bookend)
    - last_sentence (bookend)  
    - raw_markdown content
    
    Example:
        >>> fallback = BM25Fallback(toc)
        >>> result = fallback.search("authentication tokens")
        >>> print(result.top_entry.header)
    """
    
    def __init__(self, toc: TableOfContents) -> None:
        self.toc = toc
        self._index: BM25Okapi | None = None
        self._entry_map: list[ToCEntry] = []
        self._build_index()
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
    
    def _build_index(self) -> None:
        """Build BM25 index from ToC entries."""
        corpus = []
        self._entry_map = []
        
        for entry in self.toc.entries:
            doc_text = " ".join([
                entry.first_sentence,
                entry.last_sentence,
                entry.header,
                entry.dense_signpost,
                entry.raw_markdown[:1000],
            ])
            tokens = self._tokenize(doc_text)
            corpus.append(tokens)
            self._entry_map.append(entry)
        
        if corpus:
            self._index = BM25Okapi(corpus)
            logger.info(f"BM25 index built with {len(corpus)} documents")
        else:
            logger.warning("No documents to index")
    
    def search(self, query: str, top_k: int = 5) -> BM25Result:
        """Search using BM25."""
        if not self._index or not self._entry_map:
            return BM25Result(query=query, matches=[])
        
        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)
        
        scored_entries = list(zip(self._entry_map, scores))
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        top_matches = [(entry, score) for entry, score in scored_entries[:top_k] if score > 0]
        
        logger.debug(f"BM25 search: {len(top_matches)} matches for '{query[:30]}...'")
        
        return BM25Result(query=query, matches=top_matches)
    
    def rebuild_index(self) -> None:
        """Rebuild the BM25 index."""
        self._build_index()
