"""Memory-efficient Bloom Filter for URL deduplication.

Uses mmh3 for fast hashing and bitarray for compact storage.
Designed for millions of URLs with configurable false positive rate.
"""

import math
from typing import Iterable

import mmh3
from bitarray import bitarray

from vnull.core.config import settings
from vnull.core.logging import get_logger

logger = get_logger(__name__)


class BloomFilter:
    """Probabilistic data structure for set membership testing.
    
    Space-efficient with tunable false positive rate.
    No false negatives - if it says "not seen", it's definitely not seen.
    
    Example:
        >>> bf = BloomFilter(expected_items=100000, fp_rate=0.01)
        >>> bf.add("https://example.com/page1")
        >>> "https://example.com/page1" in bf
        True
        >>> "https://example.com/page2" in bf
        False
    """
    
    def __init__(
        self,
        expected_items: int | None = None,
        fp_rate: float | None = None,
    ) -> None:
        """Initialize Bloom filter.
        
        Args:
            expected_items: Expected number of items. Defaults to settings.
            fp_rate: Desired false positive rate. Defaults to settings.
        """
        self.expected_items = expected_items or settings.bloom_filter_size
        self.fp_rate = fp_rate or settings.bloom_filter_fp_rate
        
        # Calculate optimal size and hash count
        self.size = self._optimal_size(self.expected_items, self.fp_rate)
        self.hash_count = self._optimal_hash_count(self.size, self.expected_items)
        
        # Initialize bit array
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)
        
        self._count = 0
        
        logger.info(
            "Bloom filter initialized",
            size_bits=self.size,
            size_mb=round(self.size / 8 / 1024 / 1024, 2),
            hash_count=self.hash_count,
            expected_items=self.expected_items,
            fp_rate=self.fp_rate,
        )
    
    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size.
        
        m = -(n * ln(p)) / (ln(2)^2)
        """
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)
    
    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions.
        
        k = (m/n) * ln(2)
        """
        k = (m / n) * math.log(2)
        return max(1, int(k))
    
    def _get_hash_indices(self, item: str) -> list[int]:
        """Generate hash indices for an item using double hashing.
        
        Uses mmh3 with two seeds, then combines them for k hashes.
        This is more efficient than computing k independent hashes.
        """
        h1 = mmh3.hash(item, seed=0, signed=False)
        h2 = mmh3.hash(item, seed=h1 & 0xFFFFFFFF, signed=False)
        
        indices = []
        for i in range(self.hash_count):
            # Double hashing: h(i) = h1 + i*h2
            combined = (h1 + i * h2) % self.size
            indices.append(combined)
        
        return indices
    
    def add(self, item: str) -> bool:
        """Add an item to the filter.
        
        Args:
            item: String item to add (typically a URL).
            
        Returns:
            True if item was probably new, False if probably existed.
        """
        indices = self._get_hash_indices(item)
        is_new = not all(self.bit_array[i] for i in indices)
        
        for i in indices:
            self.bit_array[i] = 1
        
        if is_new:
            self._count += 1
        
        return is_new
    
    def __contains__(self, item: str) -> bool:
        """Check if item might be in the filter.
        
        Args:
            item: String item to check.
            
        Returns:
            True if item might exist (could be false positive).
            False if item definitely doesn't exist.
        """
        indices = self._get_hash_indices(item)
        return all(self.bit_array[i] for i in indices)
    
    def add_many(self, items: Iterable[str]) -> int:
        """Add multiple items efficiently.
        
        Args:
            items: Iterable of string items.
            
        Returns:
            Count of new items added.
        """
        new_count = 0
        for item in items:
            if self.add(item):
                new_count += 1
        return new_count
    
    @property
    def count(self) -> int:
        """Approximate count of items added."""
        return self._count
    
    @property
    def current_fp_rate(self) -> float:
        """Estimate current false positive rate based on fill ratio."""
        bits_set = self.bit_array.count(1)
        fill_ratio = bits_set / self.size
        return fill_ratio ** self.hash_count
    
    def clear(self) -> None:
        """Reset the filter."""
        self.bit_array.setall(0)
        self._count = 0
    
    def __len__(self) -> int:
        """Return approximate item count."""
        return self._count
    
    def __repr__(self) -> str:
        return f"BloomFilter(items={self._count}, size={self.size}, fp_rate={self.current_fp_rate:.4f})"
