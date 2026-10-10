"""Sync check: in-memory Qdrant + fake embedder, no GPU. Run: uv run python -m rag.test_store"""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient

from rag import store

store.client = lambda c=QdrantClient(":memory:"): c
store.embedder = lambda: DeterministicFakeEmbedding(size=768)
doc = lambda src, text: Document(page_content=text, metadata={"source": src})

a, b, tab = doc("p1", "alpha"), doc("p2", "beta"), doc("p1", "same tab")
assert store.sync("t", [a, b, tab, tab]) == {"added": 4, "deleted": 0, "total": 4}   # duplicate on one page kept
assert store.sync("t", [a, b, tab, tab]) == {"added": 0, "deleted": 0, "total": 4}   # no change -> no embedding
assert store.sync("t", [a, doc("p2", "beta v2"), tab]) == {"added": 1, "deleted": 2, "total": 3}   # edit + one copy removed
assert store.sync("t", [b]) == {"added": 1, "deleted": 3, "total": 1}               # pages removed
assert store.client().count("t").count == 1
print("ok")
