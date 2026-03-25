"""Table of Contents (ToC) builder for the index.

Builds lightweight JSON index containing:
- chunk_id
- dense_signpost  
- first_sentence (bookend)
- last_sentence (bookend)
- raw_markdown
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vnull.core.config import settings
from vnull.core.logging import get_logger
from vnull.indexer.chunker import MarkdownChunk
from vnull.indexer.signpost_generator import Signpost

logger = get_logger(__name__)


@dataclass
class ToCEntry:
    """Single entry in the Table of Contents."""
    chunk_id: str
    dense_signpost: str
    first_sentence: str
    last_sentence: str
    raw_markdown: str
    header: str
    header_level: int
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    source_url: str | None = None
    core_theme: str = ""
    key_entities: list[str] = field(default_factory=list)
    questions_answered: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "dense_signpost": self.dense_signpost,
            "first_sentence": self.first_sentence,
            "last_sentence": self.last_sentence,
            "header": self.header,
            "header_level": self.header_level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "source_url": self.source_url,
            "core_theme": self.core_theme,
            "key_entities": self.key_entities,
            "questions_answered": self.questions_answered,
            "raw_markdown": self.raw_markdown,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToCEntry":
        return cls(**data)


@dataclass
class TableOfContents:
    """Complete Table of Contents index."""
    entries: list[ToCEntry]
    created_at: datetime
    source_name: str
    version: str = "1.0"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_name": self.source_name,
            "created_at": self.created_at.isoformat(),
            "entry_count": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableOfContents":
        entries = [ToCEntry.from_dict(e) for e in data["entries"]]
        return cls(
            entries=entries,
            created_at=datetime.fromisoformat(data["created_at"]),
            source_name=data["source_name"],
            version=data.get("version", "1.0"),
        )
    
    def get_entry(self, chunk_id: str) -> ToCEntry | None:
        """Get entry by chunk ID."""
        for entry in self.entries:
            if entry.chunk_id == chunk_id:
                return entry
        return None
    
    def get_signposts_only(self) -> list[dict[str, str]]:
        """Get minimal signpost data for LLM routing."""
        return [
            {
                "chunk_id": e.chunk_id,
                "signpost": e.dense_signpost,
                "header": e.header,
            }
            for e in self.entries
        ]
    
    def get_parent(self, chunk_id: str) -> ToCEntry | None:
        """Get parent entry for a chunk."""
        entry = self.get_entry(chunk_id)
        if entry and entry.parent_id:
            return self.get_entry(entry.parent_id)
        return None
    
    def get_children(self, chunk_id: str) -> list[ToCEntry]:
        """Get child entries for a chunk."""
        entry = self.get_entry(chunk_id)
        if not entry:
            return []
        return [self.get_entry(cid) for cid in entry.children_ids if self.get_entry(cid)]


class ToCBuilder:
    """Build Table of Contents from chunks and signposts.
    
    Example:
        >>> builder = ToCBuilder()
        >>> toc = builder.build(chunks, signposts, "my-docs")
        >>> builder.save(toc, Path("index/my-docs.json"))
    """
    
    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize builder.
        
        Args:
            output_dir: Directory for saving ToC files.
        """
        self.output_dir = output_dir or settings.index_dir
    
    def build(
        self,
        chunks: list[MarkdownChunk],
        signposts: list[Signpost],
        source_name: str,
    ) -> TableOfContents:
        """Build ToC from chunks and signposts.
        
        Args:
            chunks: List of Markdown chunks.
            signposts: List of signposts (must match chunks by chunk_id).
            source_name: Name for this index.
            
        Returns:
            TableOfContents object.
        """
        signpost_map = {s.chunk_id: s for s in signposts}
        
        entries = []
        for chunk in chunks:
            signpost = signpost_map.get(chunk.chunk_id)
            
            entry = ToCEntry(
                chunk_id=chunk.chunk_id,
                dense_signpost=signpost.signpost if signpost else "",
                first_sentence=chunk.first_sentence,
                last_sentence=chunk.last_sentence,
                raw_markdown=chunk.content,
                header=chunk.header,
                header_level=chunk.header_level,
                parent_id=chunk.parent_id,
                children_ids=chunk.children_ids,
                source_url=chunk.source_url,
                core_theme=signpost.core_theme if signpost else "",
                key_entities=signpost.key_entities if signpost else [],
                questions_answered=signpost.questions_answered if signpost else [],
            )
            entries.append(entry)
        
        toc = TableOfContents(
            entries=entries,
            created_at=datetime.now(timezone.utc),
            source_name=source_name,
        )
        
        logger.info(
            "ToC built",
            source_name=source_name,
            entries=len(entries),
        )
        
        return toc
    
    def save(self, toc: TableOfContents, filepath: Path | None = None) -> Path:
        """Save ToC to JSON file.
        
        Args:
            toc: TableOfContents to save.
            filepath: Output path. Defaults to output_dir/source_name.json.
            
        Returns:
            Path to saved file.
        """
        if filepath is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / f"{toc.source_name}.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(toc.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info("ToC saved", path=str(filepath))
        return filepath
    
    def load(self, filepath: Path) -> TableOfContents:
        """Load ToC from JSON file.
        
        Args:
            filepath: Path to JSON file.
            
        Returns:
            TableOfContents object.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        toc = TableOfContents.from_dict(data)
        logger.info("ToC loaded", path=str(filepath), entries=len(toc.entries))
        return toc
    
    def load_all(self, directory: Path | None = None) -> list[TableOfContents]:
        """Load all ToC files from directory.
        
        Args:
            directory: Directory to scan. Defaults to output_dir.
            
        Returns:
            List of TableOfContents objects.
        """
        directory = directory or self.output_dir
        tocs = []
        
        for filepath in directory.glob("*.json"):
            try:
                toc = self.load(filepath)
                tocs.append(toc)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
        
        return tocs
    
    def merge(self, tocs: list[TableOfContents], name: str) -> TableOfContents:
        """Merge multiple ToCs into one.
        
        Args:
            tocs: List of ToCs to merge.
            name: Name for merged ToC.
            
        Returns:
            Merged TableOfContents.
        """
        all_entries = []
        for toc in tocs:
            all_entries.extend(toc.entries)
        
        return TableOfContents(
            entries=all_entries,
            created_at=datetime.now(timezone.utc),
            source_name=name,
        )
