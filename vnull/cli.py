"""CLI interface for Zero-Null RAG."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from vnull.core.config import settings
from vnull.core.logging import configure_logging

app = typer.Typer(name="vnull", help="Zero-Null Vectorless RAG System")
console = Console()


@app.command()
def crawl(
    url: str = typer.Argument(..., help="Seed URL to crawl"),
    depth: int = typer.Option(2, "--depth", "-d", help="Max crawl depth"),
    pages: int = typer.Option(50, "--pages", "-p", help="Max pages to crawl"),
    js: bool = typer.Option(False, "--js", "-j", help="Render JavaScript"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Crawl a website and save HTML files."""
    configure_logging()
    output_dir = output or settings.raw_dir
    
    async def run_crawl():
        from vnull.crawler import AsyncCrawler, JSRenderer
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task(f"Crawling {url}...", total=None)
            
            if js:
                async with JSRenderer() as renderer:
                    result = await renderer.render(url)
                    if result.is_success:
                        filepath = output_dir / "rendered.html"
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        filepath.write_text(result.content)
                        console.print(f"[green]Saved:[/green] {filepath}")
            else:
                async with AsyncCrawler() as crawler:
                    count = 0
                    async for result in crawler.crawl(url, max_depth=depth, max_pages=pages):
                        if result.is_success:
                            await crawler.save_result(result, output_dir)
                            count += 1
                            progress.update(task, description=f"Crawled {count} pages...")
                    
                    console.print(f"[green]Complete:[/green] {count} pages saved to {output_dir}")
    
    asyncio.run(run_crawl())


@app.command()
def convert(
    input_dir: Path = typer.Argument(..., help="Directory with HTML files"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Convert HTML files to Markdown using LLM."""
    configure_logging()
    output = output_dir or settings.markdown_dir
    
    async def run_convert():
        from vnull.parser import MarkdownConverter
        
        converter = MarkdownConverter()
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Converting...", total=None)
            results = await converter.convert_directory(input_dir, output)
            console.print(f"[green]Converted:[/green] {len(results)} files to {output}")
    
    asyncio.run(run_convert())


@app.command()
def index(
    name: str = typer.Argument(..., help="Index name"),
    input_dir: Optional[Path] = typer.Option(None, "--input", "-i", help="Markdown directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path"),
):
    """Build ToC index from Markdown files."""
    configure_logging()
    md_dir = input_dir or settings.markdown_dir
    
    async def run_index():
        from vnull.indexer import MarkdownChunker, SignpostGenerator, ToCBuilder
        
        chunker = MarkdownChunker()
        generator = SignpostGenerator()
        builder = ToCBuilder()
        
        all_chunks = []
        for md_file in md_dir.glob("*.md"):
            chunks = chunker.chunk_file(str(md_file))
            all_chunks.extend(chunks)
        
        console.print(f"[blue]Chunked:[/blue] {len(all_chunks)} chunks from {md_dir}")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Generating signposts...", total=len(all_chunks))
            signposts = await generator.generate_batch(all_chunks)
            progress.update(task, completed=len(all_chunks))
        
        toc = builder.build(all_chunks, signposts, name)
        out_path = output or (settings.index_dir / f"{name}.json")
        builder.save(toc, out_path)
        
        console.print(f"[green]Index saved:[/green] {out_path}")
    
    asyncio.run(run_index())


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    index_path: Optional[Path] = typer.Option(None, "--index", "-i", help="Index JSON file"),
):
    """Query the RAG system."""
    configure_logging()
    
    async def run_query():
        from vnull.retrieval import RetrievalOrchestrator
        from vnull.indexer import ToCBuilder
        
        if index_path:
            toc = ToCBuilder().load(index_path)
        else:
            toc_files = list(settings.index_dir.glob("*.json"))
            if not toc_files:
                console.print("[red]No index files found. Run 'vnull index' first.[/red]")
                raise typer.Exit(1)
            builder = ToCBuilder()
            tocs = [builder.load(f) for f in toc_files]
            toc = builder.merge(tocs, "query") if len(tocs) > 1 else tocs[0]
        
        orch = RetrievalOrchestrator(toc=toc)
        
        with console.status("Thinking..."):
            result = await orch.retrieve(question)
        
        console.print(f"\n[bold]Answer:[/bold]\n{result.answer}")
        console.print(f"\n[dim]Layer: {result.layer_used}[/dim]")
    
    asyncio.run(run_query())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the API server."""
    import uvicorn
    configure_logging()
    console.print(f"[green]Starting server at http://{host}:{port}[/green]")
    uvicorn.run("vnull.api.server:app", host=host, port=port, reload=reload)


@app.command()
def pipeline(
    url: str = typer.Argument(..., help="URL to process"),
    name: str = typer.Option("default", "--name", "-n", help="Index name"),
    depth: int = typer.Option(2, "--depth", "-d", help="Crawl depth"),
):
    """Run full pipeline: crawl -> convert -> index."""
    configure_logging()
    
    async def run_pipeline():
        from vnull.crawler import AsyncCrawler
        from vnull.parser import MarkdownConverter
        from vnull.indexer import MarkdownChunker, SignpostGenerator, ToCBuilder
        
        console.print(f"[bold]Step 1/3:[/bold] Crawling {url}")
        async with AsyncCrawler() as crawler:
            count = 0
            async for result in crawler.crawl(url, max_depth=depth, max_pages=100):
                if result.is_success:
                    await crawler.save_result(result)
                    count += 1
        console.print(f"[green]Crawled {count} pages[/green]")
        
        console.print(f"[bold]Step 2/3:[/bold] Converting to Markdown")
        converter = MarkdownConverter()
        results = await converter.convert_directory(settings.raw_dir)
        console.print(f"[green]Converted {len(results)} files[/green]")
        
        console.print(f"[bold]Step 3/3:[/bold] Building index")
        chunker = MarkdownChunker()
        generator = SignpostGenerator()
        builder = ToCBuilder()
        
        all_chunks = []
        for md_file in settings.markdown_dir.glob("*.md"):
            chunks = chunker.chunk_file(str(md_file))
            all_chunks.extend(chunks)
        
        signposts = await generator.generate_batch(all_chunks)
        toc = builder.build(all_chunks, signposts, name)
        out_path = builder.save(toc)
        
        console.print(f"[green bold]Pipeline complete![/green bold]")
        console.print(f"Index: {out_path}")
        console.print(f"Chunks: {len(all_chunks)}")
    
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    app()
