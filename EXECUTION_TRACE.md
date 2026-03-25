Zero-Null Vectorless RAG - Complete Execution Trace

This document provides a comprehensive walkthrough of the entire pipeline execution, from crawling to querying.

Project Structure (Verified)

Zero-Null Vectorless RAG - Complete Execution Trace

v-less/
├── vnull/
│   ├── __init__.py
│   ├── cli.py                      # CLI commands (crawl, convert, index, query, serve, pipeline)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Pydantic settings (LLM URL, tokens, paths)
│   │   ├── logging.py              # Structlog configuration
│   │   └── llm_client.py           # OpenAI SDK wrapper for llama.cpp
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── bloom_filter.py         # URL deduplication (mmh3 + bitarray)
│   │   ├── async_crawler.py        # aiohttp concurrent crawler
│   │   └── js_renderer.py          # Playwright stealth renderer
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── dom_pruner.py           # BeautifulSoup boilerplate removal
│   │   ├── html_splitter.py        # Token-aware splitting (Qwen tokenizer)
│   │   └── markdown_converter.py   # LLM-driven HTML→Markdown
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── chunker.py              # Header-boundary chunking + bookends
│   │   ├── signpost_generator.py   # Dense signpost generation via LLM
│   │   └── toc_builder.py          # JSON ToC builder
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── deep_sieve.py           # Layer 1: Query deconstruction
│   │   ├── toc_router.py           # Layer 2: Signpost matching
│   │   ├── explorer.py             # Layer 3: Multi-path + MCTS-lite
│   │   ├── bm25_fallback.py        # Layer 4: Lexical fallback
│   │   └── orchestrator.py         # Multi-layer orchestrator
│   └── api/
│       ├── __init__.py
│       ├── server.py               # FastAPI + streaming
│       └── schemas.py              # Pydantic models
├── tests/
│   ├── conftest.py
│   ├── test_bloom_filter.py
│   ├── test_dom_pruner.py
│   ├── test_chunker.py
│   ├── test_toc_builder.py
│   └── test_bm25_fallback.py
├── scripts/setup.sh
├── data/                           # Runtime data
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
