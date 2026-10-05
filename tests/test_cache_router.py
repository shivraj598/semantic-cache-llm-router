import os
from app import config, cache, router, rag

def test_normalize():
    assert cache.normalize_query("  What is Python?  ") == "what is python"
    assert cache.normalize_query("Hello!!!") == "hello"

def test_is_cacheable():
    assert cache._is_cacheable("what is python") is True
    assert cache._is_cacheable("ERROR: fail") is False
    assert cache._is_cacheable("hi") is False  # too short
    assert cache._is_cacheable("my username is alice") is False

def test_router_basic():
    r = router.classify("What is Python?")
    assert r["route"] in ("small", "large", "simple", "complex") or "complexity" in r
    r = router.classify("Why does Python have GIL and how does it compare to Go?")
    assert r["complexity"] >= 2 or r["route"] == "complex"

def test_weak_answer_detection():
    assert router.is_weak_answer("") is True
    assert router.is_weak_answer("I don't know") is True
    assert router.is_weak_answer("Based on context... [1]") is False
    assert router.is_weak_answer("Based on context with no citations") is True
