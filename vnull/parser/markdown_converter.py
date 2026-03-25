"""LLM-driven HTML to Markdown conversion.

Uses local LLM to translate pruned HTML into clean, semantic Markdown.
DO NOT use html2text or markitdown - this is LLM-native conversion.
"""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from vnull.core.config import settings
from vnull.core.llm_client import LLMClient
from vnull.core.logging import get_logger
from vnull.parser.dom_pruner import DOMPruner
from vnull.parser.html_splitter import HTMLSplitter, HTMLChunk

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a precise HTML-to-Markdown converter. Your task is to translate HTML into clean, semantic Markdown.

RULES:
1. Preserve ALL text content exactly - do not summarize or omit anything
2. Convert HTML tables to proper Markdown tables with |---| separators
3. Preserve code blocks with appropriate language tags (```python, ```javascript, etc.)
4. Convert headings: <h1> to #, <h2> to ##, etc.
5. Convert lists properly: <ul>/<li> to -, <ol>/<li> to 1. 2. 3.
6. Convert links: <a href="url">text</a> to [text](url)
7. Convert images: <img src="url" alt="text"> to ![text](url)
8. Convert emphasis: <strong>/<b> to **, <em>/<i> to *
9. Remove all HTML tags from output - output ONLY Markdown
10. Preserve paragraph breaks with blank lines
11. Do NOT add any commentary or explanations - output ONLY the converted Markdown

IMPORTANT: Output raw Markdown only. No code fences around the entire output."""


@dataclass
class ConversionResult:
    """Result of HTML to Markdown conversion."""
    markdown: str
    source_url: str | None
    original_html_size: int
    markdown_size: int
    chunks_processed: int
    
    @property
    def compression_ratio(self) -> float:
        if self.original_html_size == 0:
            return 0.0
        return self.markdown_size / self.original_html_size


class MarkdownConverter:
    """Convert HTML to Markdown using local LLM.
    
    Pipeline:
    1. Prune DOM to remove boilerplate
    2. Split if exceeds token limit
    3. Send each chunk to LLM for conversion
    4. Merge results
    
    Example:
        >>> converter = MarkdownConverter()
        >>> result = await converter.convert(html_content)
        >>> print(result.markdown)
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        pruner: DOMPruner | None = None,
        splitter: HTMLSplitter | None = None,
    ) -> None:
        """Initialize converter.
        
        Args:
            llm_client: LLM client instance.
            pruner: DOM pruner instance.
            splitter: HTML splitter instance.
        """
        self.llm = llm_client or LLMClient()
        self.pruner = pruner or DOMPruner()
        self.splitter = splitter or HTMLSplitter()
    
    async def _convert_chunk(self, chunk: HTMLChunk) -> str:
        """Convert a single HTML chunk to Markdown."""
        prompt = f"""Convert this HTML to Markdown:

{chunk.content}"""
        
        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        )
        
        markdown = response.content.strip()
        markdown = re.sub(r"^```markdown\s*", "", markdown)
        markdown = re.sub(r"\s*```$", "", markdown)
        
        return markdown
    
    async def convert(
        self,
        html: str,
        source_url: str | None = None,
        prune: bool = True,
    ) -> ConversionResult:
        """Convert HTML to Markdown.
        
        Args:
            html: Raw HTML content.
            source_url: Optional source URL for metadata.
            prune: Whether to prune DOM first.
            
        Returns:
            ConversionResult with Markdown and metadata.
        """
        original_size = len(html)
        
        if prune:
            prune_result = self.pruner.prune(html)
            html = prune_result.pruned_html
            logger.debug(
                "HTML pruned before conversion",
                reduction_percent=round(prune_result.reduction_percent, 1),
            )
        
        chunks = self.splitter.split(html)
        
        logger.info(
            "Converting HTML to Markdown",
            chunks=len(chunks),
            source_url=source_url,
        )
        
        if len(chunks) == 1:
            markdown = await self._convert_chunk(chunks[0])
        else:
            markdown_parts = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Converting chunk {i+1}/{len(chunks)}")
                part = await self._convert_chunk(chunk)
                markdown_parts.append(part)
                
                if i < len(chunks) - 1:
                    await self.llm.flush_kv_cache()
            
            markdown = "\n\n".join(markdown_parts)
        
        markdown = self._clean_markdown(markdown)
        
        return ConversionResult(
            markdown=markdown,
            source_url=source_url,
            original_html_size=original_size,
            markdown_size=len(markdown),
            chunks_processed=len(chunks),
        )
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up converted Markdown."""
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        markdown = re.sub(r"[ \t]+$", "", markdown, flags=re.MULTILINE)
        markdown = re.sub(r"^[ \t]+", "", markdown, flags=re.MULTILINE)
        lines = markdown.split("\n")
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank
        
        return "\n".join(cleaned_lines).strip()
    
    async def convert_file(
        self,
        input_path: Path,
        output_path: Path | None = None,
    ) -> ConversionResult:
        """Convert HTML file to Markdown file.
        
        Args:
            input_path: Path to HTML file.
            output_path: Path for output Markdown file.
            
        Returns:
            ConversionResult.
        """
        html = input_path.read_text(encoding="utf-8")
        
        url_match = re.search(r"<!-- URL: (.+?) -->", html)
        source_url = url_match.group(1) if url_match else None
        
        result = await self.convert(html, source_url=source_url)
        
        if output_path is None:
            output_path = input_path.with_suffix(".md")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        header = ""
        if source_url:
            header = f"<!-- Source: {source_url} -->\n\n"
        
        output_path.write_text(header + result.markdown, encoding="utf-8")
        
        logger.info(
            "Converted file",
            input=str(input_path),
            output=str(output_path),
        )
        
        return result
    
    async def convert_directory(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        pattern: str = "*.html",
    ) -> list[ConversionResult]:
        """Convert all HTML files in directory.
        
        Args:
            input_dir: Directory with HTML files.
            output_dir: Output directory for Markdown.
            pattern: Glob pattern for HTML files.
            
        Returns:
            List of ConversionResults.
        """
        input_dir = Path(input_dir)
        output_dir = output_dir or settings.markdown_dir
        
        html_files = list(input_dir.glob(pattern))
        logger.info(f"Found {len(html_files)} HTML files to convert")
        
        results = []
        for html_file in html_files:
            output_file = output_dir / html_file.with_suffix(".md").name
            try:
                result = await self.convert_file(html_file, output_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to convert {html_file}: {e}")
        
        return results
