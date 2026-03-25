"""Tests for ToC builder."""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from vnull.indexer.toc_builder import ToCBuilder, ToCEntry, TableOfContents
from vnull.indexer.chunker import MarkdownChunk
from vnull.indexer.signpost_generator import Signpost


def test_build_toc():
    chunks = [MarkdownChunk(
        chunk_id="test1",
        content="# Test\n\nContent",
        header="Test",
        header_level=1,
        first_sentence="Content",
        last_sentence="Content",
    )]
    signposts = [Signpost(
        chunk_id="test1",
        signpost="[Test] + [content] + [What is test?]",
        core_theme="Test",
        key_entities=["content"],
        questions_answered=["What is test?"],
        token_count=10,
    )]
    
    builder = ToCBuilder()
    toc = builder.build(chunks, signposts, "test-source")
    
    assert len(toc.entries) == 1
    assert toc.entries[0].chunk_id == "test1"


def test_save_load_toc():
    toc = TableOfContents(
        entries=[ToCEntry(
            chunk_id="x",
            dense_signpost="sig",
            first_sentence="first",
            last_sentence="last",
            raw_markdown="# X\n\nContent",
            header="X",
            header_level=1,
        )],
        created_at=datetime.now(timezone.utc),
        source_name="test",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = ToCBuilder(Path(tmpdir))
        path = builder.save(toc)
        loaded = builder.load(path)
        
        assert loaded.source_name == "test"
        assert len(loaded.entries) == 1
