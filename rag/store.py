"""Embedder + Qdrant collections, kept in sync with the corpus by content-addressed chunk IDs."""
import atexit
import os
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@lru_cache
def embedder() -> HuggingFaceEmbeddings:
    e = HuggingFaceEmbeddings(model_name=os.environ["EMBED_MODEL"], model_kwargs={"device": "cuda"},
                              encode_kwargs={"prompt_name": "document", "batch_size": 32},
                              query_encode_kwargs={"prompt_name": "query"})
    e._client.max_seq_length = 8192   # loads unset; cap at the model card limit
    return e


@lru_cache
def client() -> QdrantClient:
    """Qdrant server if QDRANT_URL is set (compose.yml; many processes, HNSW index), else embedded local mode
    (exact brute-force search, one process at a time)."""
    url = os.environ.get("QDRANT_URL")
    c = QdrantClient(url=url) if url else QdrantClient(path=str(ROOT / "data/qdrant"))
    atexit.register(c.close)   # else __del__ runs during interpreter teardown and prints an ImportError
    return c


def chunk_ids(chunks) -> list[str]:
    """ID = hash of (page, text, nth copy on that page). Same content -> same ID, so only changed chunks re-embed.
    The copy counter keeps identical chunks on one page (provider tabs) as separate points."""
    seen = Counter()
    ids = []
    for d in chunks:
        key = (d.metadata["source"], d.page_content)
        seen[key] += 1
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{key[0]}\n{seen[key]}\n{key[1]}")))
    return ids


def sync(name: str, chunks) -> dict:
    """Make collection `name` hold exactly `chunks`: embed new IDs, delete IDs no longer produced."""
    c = client()
    if not c.collection_exists(name):
        c.create_collection(name, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
    have, offset = set(), None
    while True:
        pts, offset = c.scroll(name, limit=10_000, offset=offset, with_payload=False)
        have |= {str(p.id) for p in pts}
        if offset is None:
            break
    want = dict(zip(chunk_ids(chunks), chunks))
    new = [i for i in want if i not in have]
    gone = list(have - want.keys())
    if gone:
        c.delete(name, points_selector=models.PointIdsList(points=gone))
    db = QdrantVectorStore(client=c, collection_name=name, embedding=embedder())
    for i in range(0, len(new), 1000):
        db.add_documents([want[j] for j in new[i:i + 1000]], ids=new[i:i + 1000])
    if new:
        import torch
        torch.cuda.empty_cache()   # bulk embedding leaves its peak reserved; free it or later queries can't get cuBLAS memory (8 GB GPU)
    return {"added": len(new), "deleted": len(gone), "total": len(want)}


def get_store(name: str, chunks) -> QdrantVectorStore:
    stats = sync(name, chunks)
    print(f"[{name}] sync {stats}")
    return QdrantVectorStore(client=client(), collection_name=name, embedding=embedder())
