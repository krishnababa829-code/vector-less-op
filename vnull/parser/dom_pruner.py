"""Aggressive DOM pruning to remove boilerplate elements.

Strips scripts, styles, navigation, footers, and other non-content elements
before passing to LLM for conversion.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from vnull.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PruneResult:
    """Result of DOM pruning operation."""
    original_size: int
    pruned_size: int
    pruned_html: str
    elements_removed: dict[str, int]
    
    @property
    def reduction_percent(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (1 - self.pruned_size / self.original_size) * 100


class DOMPruner:
    """Aggressive DOM pruner for removing boilerplate.
    
    Removes:
    - <script>, <style>, <noscript> tags
    - <nav>, <footer>, <header> (optional)
    - <svg>, <canvas>, <iframe>
    - Comments and hidden elements
    - Empty containers
    - Ad-related elements
    
    Example:
        >>> pruner = DOMPruner()
        >>> result = pruner.prune(html_content)
        >>> print(f"Reduced by {result.reduction_percent:.1f}%")
    """
    
    ALWAYS_REMOVE = {
        "script", "style", "noscript", "svg", "canvas", "iframe",
        "video", "audio", "source", "track", "embed", "object",
        "param", "map", "area",
    }
    
    BOILERPLATE_TAGS = {
        "nav", "footer", "aside", "header",
    }
    
    AD_PATTERNS = [
        r"ad[-_]?", r"ads[-_]?", r"advert", r"banner", r"sponsor",
        r"promo", r"sidebar", r"widget", r"popup", r"modal",
        r"cookie", r"gdpr", r"consent", r"newsletter", r"subscribe",
        r"social[-_]?share", r"share[-_]?button", r"comment[-_]?section",
        r"related[-_]?post", r"recommended", r"trending",
    ]
    
    def __init__(
        self,
        remove_nav: bool = True,
        remove_footer: bool = True,
        remove_header: bool = False,
        remove_aside: bool = True,
        remove_ads: bool = True,
        remove_empty: bool = True,
        min_text_length: int = 20,
    ) -> None:
        """Initialize pruner with configuration.
        
        Args:
            remove_nav: Remove <nav> elements.
            remove_footer: Remove <footer> elements.
            remove_header: Remove <header> elements.
            remove_aside: Remove <aside> elements.
            remove_ads: Remove ad-related elements by class/id.
            remove_empty: Remove empty container elements.
            min_text_length: Minimum text length to keep element.
        """
        self.remove_nav = remove_nav
        self.remove_footer = remove_footer
        self.remove_header = remove_header
        self.remove_aside = remove_aside
        self.remove_ads = remove_ads
        self.remove_empty = remove_empty
        self.min_text_length = min_text_length
        
        self._ad_pattern = re.compile(
            "|".join(self.AD_PATTERNS),
            re.IGNORECASE
        )
    
    def _should_remove_tag(self, tag: Tag) -> tuple[bool, str]:
        """Determine if a tag should be removed.
        
        Returns:
            Tuple of (should_remove, reason).
        """
        tag_name = tag.name.lower() if tag.name else ""
        
        if tag_name in self.ALWAYS_REMOVE:
            return True, f"always_remove:{tag_name}"
        
        if tag_name == "nav" and self.remove_nav:
            return True, "nav"
        if tag_name == "footer" and self.remove_footer:
            return True, "footer"
        if tag_name == "header" and self.remove_header:
            return True, "header"
        if tag_name == "aside" and self.remove_aside:
            return True, "aside"
        
        if self.remove_ads:
            classes = " ".join(tag.get("class", []))
            tag_id = tag.get("id", "")
            combined = f"{classes} {tag_id}"
            
            if self._ad_pattern.search(combined):
                return True, "ad_pattern"
        
        if tag.get("hidden") or tag.get("aria-hidden") == "true":
            return True, "hidden"
        
        style = tag.get("style", "")
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
            return True, "hidden_style"
        
        return False, ""
    
    def _is_empty_container(self, tag: Tag) -> bool:
        """Check if element is an empty container."""
        if tag.name in {"br", "hr", "img", "input"}:
            return False
        
        text = tag.get_text(strip=True)
        return len(text) < self.min_text_length
    
    def prune(self, html: str) -> PruneResult:
        """Prune boilerplate from HTML.
        
        Args:
            html: Raw HTML content.
            
        Returns:
            PruneResult with pruned HTML and statistics.
        """
        original_size = len(html)
        elements_removed: dict[str, int] = {}
        
        soup = BeautifulSoup(html, "lxml")
        
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()
            elements_removed["comment"] = elements_removed.get("comment", 0) + 1
        
        for tag_name in self.ALWAYS_REMOVE:
            for tag in soup.find_all(tag_name):
                tag.decompose()
                elements_removed[tag_name] = elements_removed.get(tag_name, 0) + 1
        
        for tag in soup.find_all(True):
            if not isinstance(tag, Tag):
                continue
            
            should_remove, reason = self._should_remove_tag(tag)
            if should_remove:
                tag.decompose()
                elements_removed[reason] = elements_removed.get(reason, 0) + 1
        
        if self.remove_empty:
            for _ in range(3):
                for tag in soup.find_all(["div", "span", "section", "article", "p"]):
                    if isinstance(tag, Tag) and self._is_empty_container(tag):
                        tag.decompose()
                        elements_removed["empty"] = elements_removed.get("empty", 0) + 1
        
        body = soup.find("body")
        if body:
            pruned_html = str(body)
        else:
            pruned_html = str(soup)
        
        pruned_html = re.sub(r"\n\s*\n", "\n", pruned_html)
        pruned_html = re.sub(r"  +", " ", pruned_html)
        
        logger.debug(
            "DOM pruned",
            original_size=original_size,
            pruned_size=len(pruned_html),
            reduction_percent=round((1 - len(pruned_html) / original_size) * 100, 1) if original_size > 0 else 0,
            elements_removed=elements_removed,
        )
        
        return PruneResult(
            original_size=original_size,
            pruned_size=len(pruned_html),
            pruned_html=pruned_html,
            elements_removed=elements_removed,
        )
    
    def extract_main_content(self, html: str) -> str:
        """Extract main content area if identifiable.
        
        Looks for <main>, <article>, or content-like containers.
        Falls back to full pruned body if not found.
        """
        soup = BeautifulSoup(html, "lxml")
        
        main = soup.find("main")
        if main and len(main.get_text(strip=True)) > 100:
            return str(main)
        
        article = soup.find("article")
        if article and len(article.get_text(strip=True)) > 100:
            return str(article)
        
        content_patterns = ["content", "main", "article", "post", "entry", "body"]
        for pattern in content_patterns:
            for tag in soup.find_all(["div", "section"], class_=re.compile(pattern, re.I)):
                if len(tag.get_text(strip=True)) > 200:
                    return str(tag)
            for tag in soup.find_all(["div", "section"], id=re.compile(pattern, re.I)):
                if len(tag.get_text(strip=True)) > 200:
                    return str(tag)
        
        return self.prune(html).pruned_html
