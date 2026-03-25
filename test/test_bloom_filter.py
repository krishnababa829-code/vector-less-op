"""Tests for Bloom filter."""
import pytest
from vnull.crawler.bloom_filter import BloomFilter


def test_bloom_filter_add_contains():
    bf = BloomFilter(expected_items=1000, fp_rate=0.01)
    bf.add("https://example.com/page1")
    assert "https://example.com/page1" in bf
    assert "https://example.com/page2" not in bf


def test_bloom_filter_no_false_negatives():
    bf = BloomFilter(expected_items=100, fp_rate=0.01)
    urls = [f"https://example.com/page{i}" for i in range(100)]
    for url in urls:
        bf.add(url)
    for url in urls:
        assert url in bf


def test_bloom_filter_count():
    bf = BloomFilter(expected_items=100, fp_rate=0.01)
    for i in range(50):
        bf.add(f"url{i}")
    assert bf.count == 50
