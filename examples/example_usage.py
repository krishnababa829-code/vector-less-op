"""Example usage of Zero-Null Vectorless RAG system.

This script demonstrates the complete pipeline execution:
1. Crawling a website
2. Converting HTML to Markdown
3. Building the ToC index
4. Querying the system

Requirements:
- llama.cpp server running on port 8000
- Qwen 2.5B model loaded

Usage:
    python -m examples.example_usage
"""

import asyncio
from pathlib import Path
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED_URL = "https://example.com"  # Replace with your target URL
INDEX_NAME = "example-docs"
MAX_DEPTH = 2
MAX_PAGES = 10


# ============================================================================
# STEP 1: CRAWLING
# ============================================================================

async def step1_crawl(url: str, max_depth: int = 2, max_pages: int = 10):
    """Crawl a website and save HTML files.
    
    This step:
    - Initializes AsyncCrawler with Bloom filter for deduplication
    - Fetches pages concurrently with rate limiting
    - Extracts links and follows them up to max_depth
    - Saves HTML files with metadata headers
    """
    from vnull.crawler import AsyncCrawler
    from vnull.core.config import settings
    
    print(f"\n{'='*60}")
    print("STEP 1: CRAWLING")
    print(f"{'='*60}")
    print(f"Seed URL: {url}")
    print(f"Max Depth: {max_depth}")
    print(f"Max Pages: {max_pages}")
    print(f"Output: {settings.raw_dir}")
    print()
    
    crawled_files = []
    
    async with AsyncCrawler() as crawler:
        print(f"Bloom filter initialized: {crawler.bloom}")
        print(f"Max concurrent requests: {crawler.max_concurrent}")
        print(f"Delay between requests: {crawler.delay_ms}ms")
        print()
        
        count = 0
        async for result in crawler.crawl(url, max_depth=max_depth, max_pages=max_pages):
            count += 1
            
            if result.is_success:
                filepath = await crawler.save_result(result)
                crawled_files.append(filepath)
                print(f"  [{count}] ✓ {result.url}")
                print(f"      Status: {result.status_code}, Size: {result.content_length} bytes")
                print(f"      Links found: {len(result.links)}, Depth: {result.depth}")
                print(f"      Saved: {filepath.name}")
            else:
                print(f"  [{count}] ✗ {result.url}")
                print(f"      Error: {result.error}")
            print()
    
    print(f"Crawl complete: {len(crawled_files)} pages saved")
    print(f"URLs seen by Bloom filter: {len(crawler.bloom)}")
    
    return crawled_files


# ============================================================================
# STEP 2: HTML TO MARKDOWN CONVERSION
# ============================================================================

async def step2_convert():
    """Convert HTML files to Markdown using LLM.
    
    This step:
    - Prunes DOM (removes scripts, styles, nav, footer, ads)
    - Splits large HTML at structural boundaries if > 6000 tokens
    - Sends each chunk to LLM for conversion
    - Flushes KV cache between chunks to prevent OOM
    """
    from vnull.parser import MarkdownConverter, DOMPruner
    from vnull.core.config import settings
    
    print(f"\n{'='*60}")
    print("STEP 2: HTML TO MARKDOWN CONVERSION")
    print(f"{'='*60}")
    print(f"Input: {settings.raw_dir}")
    print(f"Output: {settings.markdown_dir}")
    print()
    
    converter = MarkdownConverter()
    
    # Show what DOM pruner removes
    print("DOM Pruner configuration:")
    print(f"  - Removes: <script>, <style>, <nav>, <footer>, <svg>, <aside>")
    print(f"  - Removes elements with ad-related classes/ids")
    print(f"  - Removes hidden elements")
    print()
    
    html_files = list(settings.raw_dir.glob("*.html"))
    print(f"Found {len(html_files)} HTML files to convert")
    print()
    
    results = []
    for i, html_file in enumerate(html_files, 1):
        print(f"  [{i}/{len(html_files)}] Converting {html_file.name}...")
        
        try:
            result = await converter.convert_file(
                html_file,
                settings.markdown_dir / html_file.with_suffix(".md").name
            )
            results.append(result)
            
            print(f"      Original: {result.original_html_size} chars")
            print(f"      Markdown: {result.markdown_size} chars")
            print(f"      Compression: {result.compression_ratio:.1%}")
            print(f"      Chunks processed: {result.chunks_processed}")
        except Exception as e:
            print(f"      Error: {e}")
        print()
    
    print(f"Conversion complete: {len(results)} files converted")
    
    return results


# ============================================================================
# STEP 3: INDEXING (CHUNKING + SIGNPOST GENERATION)
# ============================================================================

async def step3_index(name: str):
    """Build ToC index from Markdown files.
    
    This step:
    - Chunks Markdown at header boundaries (#, ##, etc.)
    - Extracts bookend metadata (first_sentence, last_sentence)
    - Generates dense signposts via LLM: [Theme] + [Entities] + [Questions]
    - Builds JSON Table of Contents
    """
    from vnull.indexer import MarkdownChunker, SignpostGenerator, ToCBuilder
    from vnull.core.config import settings
    
    print(f"\n{'='*60}")
    print("STEP 3: INDEXING")
    print(f"{'='*60}")
    print(f"Input: {settings.markdown_dir}")
    print(f"Output: {settings.index_dir / f'{name}.json'}")
    print()
    
    # Step 3a: Chunk Markdown
    print("Step 3a: Chunking Markdown at header boundaries...")
    chunker = MarkdownChunker(min_chunk_size=100)
    
    all_chunks = []
    md_files = list(settings.markdown_dir.glob("*.md"))
    
    for md_file in md_files:
        chunks = chunker.chunk_file(str(md_file))
        all_chunks.extend(chunks)
        print(f"  {md_file.name}: {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    print()
    
    # Show sample chunk structure
    if all_chunks:
        sample = all_chunks[0]
        print("Sample chunk structure:")
        print(f"  chunk_id: {sample.chunk_id}")
        print(f"  header: {sample.header}")
        print(f"  header_level: {sample.header_level}")
        print(f"  first_sentence: {sample.first_sentence[:80]}...")
        print(f"  last_sentence: {sample.last_sentence[:80]}...")
        print(f"  parent_id: {sample.parent_id}")
        print(f"  content length: {len(sample.content)} chars")
        print()
    
    # Step 3b: Generate Signposts
    print("Step 3b: Generating dense signposts via LLM...")
    print("  Format: [Core Theme] + [Key Entities] + [Questions Answered]")
    print("  Max tokens per signpost: 30")
    print()
    
    generator = SignpostGenerator()
    signposts = await generator.generate_batch(all_chunks, flush_cache=True)
    
    # Show sample signposts
    print("\nSample signposts generated:")
    for sp in signposts[:3]:
        print(f"  - {sp.signpost}")
    print()
    
    # Step 3c: Build ToC
    print("Step 3c: Building Table of Contents JSON...")
    builder = ToCBuilder()
    toc = builder.build(all_chunks, signposts, name)
    
    output_path = builder.save(toc)
    
    print(f"\nToC saved: {output_path}")
    print(f"  Entries: {len(toc.entries)}")
    print(f"  Source: {toc.source_name}")
    print(f"  Created: {toc.created_at.isoformat()}")
    
    return toc


# ============================================================================
# STEP 4: QUERYING (4-LAYER RETRIEVAL)
# ============================================================================

async def step4_query(question: str):
    """Query the RAG system using 4-layer retrieval.
    
    Layers:
    1. DeepSieve: Query deconstruction with <think> scratchpad
    2. ToC Router: LLM matches query to signposts, returns chunk_ids
    3. Explorer: Extracts facts from chunks, synthesizes answer
    4. BM25 Fallback: Lexical search if routing fails
    """
    from vnull.retrieval import RetrievalOrchestrator
    from vnull.indexer import ToCBuilder
    from vnull.core.config import settings
    
    print(f"\n{'='*60}")
    print("STEP 4: QUERYING")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print()
    
    # Load ToC
    toc_files = list(settings.index_dir.glob("*.json"))
    if not toc_files:
        print("ERROR: No index files found. Run indexing first.")
        return None
    
    builder = ToCBuilder()
    tocs = [builder.load(f) for f in toc_files]
    toc = builder.merge(tocs, "query") if len(tocs) > 1 else tocs[0]
    
    print(f"Loaded ToC: {len(toc.entries)} entries")
    print()
    
    # Initialize orchestrator
    orchestrator = RetrievalOrchestrator(toc=toc)
    
    print("Executing 4-layer retrieval pipeline...")
    print()
    
    # Execute retrieval
    result = await orchestrator.retrieve(question)
    
    # Show results
    print(f"{'─'*60}")
    print("RETRIEVAL RESULT")
    print(f"{'─'*60}")
    print(f"Layer used: {result.layer_used}")
    print(f"Success: {result.success}")
    print()
    
    if result.sieve_result:
        print(f"Layer 1 (DeepSieve):")
        print(f"  Action: {result.sieve_result.action}")
        print(f"  Queries: {result.sieve_result.queries}")
        print()
    
    if result.route_result:
        print(f"Layer 2 (ToC Router):")
        print(f"  Chunk IDs: {result.route_result.chunk_ids}")
        print(f"  Confidence: {result.route_result.confidence:.2f}")
        print()
    
    if result.exploration_result:
        print(f"Layer 3 (Explorer):")
        print(f"  Chunks explored: {result.exploration_result.chunks_explored}")
        print(f"  Parent explorations: {result.exploration_result.parent_explorations}")
        print()
    
    if result.bm25_result:
        print(f"Layer 4 (BM25 Fallback):")
        print(f"  Top match: {result.bm25_result.top_entry.header if result.bm25_result.top_entry else 'None'}")
        print(f"  Score: {result.bm25_result.top_score:.2f}")
        print()
    
    print(f"{'─'*60}")
    print("ANSWER")
    print(f"{'─'*60}")
    print(result.answer)
    print()
    
    return result


# ============================================================================
# FULL PIPELINE
# ============================================================================

async def run_full_pipeline():
    """Run the complete Zero-Null RAG pipeline."""
    print("\n" + "="*60)
    print("ZERO-NULL VECTORLESS RAG - FULL PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    # Step 1: Crawl
    await step1_crawl(SEED_URL, max_depth=MAX_DEPTH, max_pages=MAX_PAGES)
    
    # Step 2: Convert
    await step2_convert()
    
    # Step 3: Index
    await step3_index(INDEX_NAME)
    
    # Step 4: Query
    await step4_query("What is this website about?")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")


# ============================================================================
# STANDALONE EXAMPLES
# ============================================================================

async def example_bloom_filter():
    """Demonstrate Bloom filter for URL deduplication."""
    from vnull.crawler import BloomFilter
    
    print("\n" + "="*60)
    print("BLOOM FILTER EXAMPLE")
    print("="*60)
    
    bf = BloomFilter(expected_items=10000, fp_rate=0.01)
    print(f"Initialized: {bf}")
    print(f"Memory: ~{bf.size / 8 / 1024:.1f} KB")
    print()
    
    # Add URLs
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]
    
    for url in urls:
        bf.add(url)
        print(f"Added: {url}")
    
    print()
    
    # Check membership
    test_urls = [
        "https://example.com/page1",  # Should be True
        "https://example.com/page4",  # Should be False
    ]
    
    for url in test_urls:
        print(f"{url} in filter: {url in bf}")
    
    print(f"\nItems added: {bf.count}")
    print(f"Current FP rate: {bf.current_fp_rate:.6f}")


async def example_dom_pruning():
    """Demonstrate DOM pruning."""
    from vnull.parser import DOMPruner
    
    print("\n" + "="*60)
    print("DOM PRUNING EXAMPLE")
    print("="*60)
    
    html = """
    <html>
    <head><title>Test</title></head>
    <body>
        <script>console.log('tracking');</script>
        <style>.ad { display: block; }</style>
        <nav>Navigation Menu</nav>
        <div class="advertisement">Buy now!</div>
        <main>
            <h1>Main Content</h1>
            <p>This is the important content we want to keep.</p>
            <p>More valuable information here.</p>
        </main>
        <aside>Sidebar content</aside>
        <footer>Copyright 2024</footer>
    </body>
    </html>
    """
    
    print(f"Original HTML: {len(html)} chars")
    print()
    
    pruner = DOMPruner(
        remove_nav=True,
        remove_footer=True,
        remove_aside=True,
        remove_ads=True,
    )
    
    result = pruner.prune(html)
    
    print(f"Pruned HTML: {result.pruned_size} chars")
    print(f"Reduction: {result.reduction_percent:.1f}%")
    print()
    print("Elements removed:")
    for element, count in result.elements_removed.items():
        print(f"  - {element}: {count}")
    print()
    print("Pruned content:")
    print(result.pruned_html[:500])


async def example_chunking():
    """Demonstrate Markdown chunking with bookends."""
    from vnull.indexer import MarkdownChunker
    
    print("\n" + "="*60)
    print("MARKDOWN CHUNKING EXAMPLE")
    print("="*60)
    
    markdown = """
# API Documentation

Welcome to our API documentation. This guide covers all endpoints.

## Authentication

All API requests require authentication. You must include an Authorization header with a valid JWT token. Tokens are obtained from the /auth/token endpoint. Tokens expire after 1 hour.

## Endpoints

The API provides several endpoints for different operations.

### GET /users

Returns a list of all users. Supports pagination via query parameters. Maximum 100 users per page.

### POST /users

Creates a new user. Requires admin privileges. Returns the created user object.
"""
    
    chunker = MarkdownChunker(min_chunk_size=50)
    chunks = chunker.chunk(markdown)
    
    print(f"Total chunks: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Header: {chunk.header} (level {chunk.header_level})")
        print(f"  Parent: {chunk.parent_id}")
        print(f"  First sentence: {chunk.first_sentence[:60]}...")
        print(f"  Last sentence: {chunk.last_sentence[:60]}...")
        print(f"  Content length: {chunk.char_count} chars")
        print()


async def example_bm25_search():
    """Demonstrate BM25 fallback search."""
    from datetime import datetime, timezone
    from vnull.retrieval import BM25Fallback
    from vnull.indexer.toc_builder import TableOfContents, ToCEntry
    
    print("\n" + "="*60)
    print("BM25 FALLBACK SEARCH EXAMPLE")
    print("="*60)
    
    # Create sample ToC
    toc = TableOfContents(
        entries=[
            ToCEntry(
                chunk_id="auth1",
                dense_signpost="[Authentication] + [OAuth2, JWT, tokens] + [How to authenticate?]",
                first_sentence="All API requests require authentication via JWT tokens.",
                last_sentence="Tokens expire after 1 hour and must be refreshed.",
                raw_markdown="## Authentication\n\nAll API requests require authentication via JWT tokens. Include the token in the Authorization header.",
                header="Authentication",
                header_level=2,
            ),
            ToCEntry(
                chunk_id="rate1",
                dense_signpost="[Rate Limiting] + [throttling, 429, limits] + [What are rate limits?]",
                first_sentence="The API enforces rate limits to ensure fair usage.",
                last_sentence="Exceeding limits returns HTTP 429 Too Many Requests.",
                raw_markdown="## Rate Limiting\n\nThe API enforces rate limits to ensure fair usage. Default limit is 100 requests per minute.",
                header="Rate Limiting",
                header_level=2,
            ),
            ToCEntry(
                chunk_id="error1",
                dense_signpost="[Error Handling] + [HTTP codes, exceptions] + [How to handle errors?]",
                first_sentence="The API uses standard HTTP status codes for errors.",
                last_sentence="Always check the error message in the response body.",
                raw_markdown="## Error Handling\n\nThe API uses standard HTTP status codes. 4xx for client errors, 5xx for server errors.",
                header="Error Handling",
                header_level=2,
            ),
        ],
        created_at=datetime.now(timezone.utc),
        source_name="example",
    )
    
    bm25 = BM25Fallback(toc)
    
    queries = [
        "How do I authenticate with JWT?",
        "What happens when I exceed rate limits?",
        "HTTP error codes",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = bm25.search(query, top_k=2)
        
        print(f"Results:")
        for entry, score in result.matches:
            print(f"  - {entry.header}: {score:.2f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
Zero-Null Vectorless RAG - Example Usage
=========================================

Available examples:
  1. full      - Run complete pipeline (crawl -> convert -> index -> query)
  2. bloom     - Bloom filter demonstration
  3. prune     - DOM pruning demonstration  
  4. chunk     - Markdown chunking demonstration
  5. bm25      - BM25 search demonstration

Usage: python -m examples.example_usage <example_name>

Note: 'full' requires llama.cpp server running on port 8000
""")
    
    if len(sys.argv) < 2:
        example = "chunk"  # Default to a simple example
    else:
        example = sys.argv[1].lower()
    
    examples = {
        "full": run_full_pipeline,
        "bloom": example_bloom_filter,
        "prune": example_dom_pruning,
        "chunk": example_chunking,
        "bm25": example_bm25_search,
    }
    
    if example in examples:
        asyncio.run(examples[example]())
    else:
        print(f"Unknown example: {example}")
        print(f"Available: {', '.join(examples.keys())}")
