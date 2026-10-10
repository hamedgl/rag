"""Embedder + persisted Chroma collections. Moves to Qdrant in Milestone 2."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@lru_cache
def embedder() -> HuggingFaceEmbeddings:
    e = HuggingFaceEmbeddings(model_name=os.environ["EMBED_MODEL"], model_kwargs={"device": "cuda"},
                              encode_kwargs={"prompt_name": "document", "batch_size": 32},
                              query_encode_kwargs={"prompt_name": "query"})
    e._client.max_seq_length = 8192   # loads unset; cap at the model card limit
    return e


def get_store(name: str, build_chunks) -> Chroma:
    """Open collection `name`; embed `build_chunks()` into it the first time.
    ponytail: "empty = build". A changed corpus or chunker needs a new name (or delete data/chroma); incremental sync is Milestone 2."""
    db = Chroma(collection_name=name, embedding_function=embedder(), persist_directory=str(ROOT / "data/chroma"),
                collection_metadata={"hnsw:space": "cosine"})
    if db._collection.count() == 0:
        chunks = build_chunks()
        for i in range(0, len(chunks), 1000):   # Chroma caps batch size
            db.add_documents(chunks[i:i + 1000])
    return db
