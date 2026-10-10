"""Retrieval pipeline: dense search (+ optional rewrite fused by RRF), then listwise LLM rerank."""
import os
import re
import time
from functools import lru_cache

import httpx
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rank_bm25 import BM25Okapi

from rag import corpus, generate

tokens = lambda s: re.findall(r"\w+", s.lower())   # \w keeps snake_case identifiers (create_agent) whole

RERANK_PROMPT = ChatPromptTemplate.from_messages([("human", """Rank documentation passages by how useful they are for answering the question.
A useful passage contains the actual answer (the fact, value, code or steps), not just the same topic.

Question: {question}

Passages:
{passages}

Reply with the numbers of the {top} most useful passages, best first, comma-separated (e.g. 3, 0, 7). Numbers only.""")])
rerank_chain = RERANK_PROMPT | generate.llm | StrOutputParser()


@lru_cache
def _llm_reranker(model: str):
    return RERANK_PROMPT | generate._chat(model) | StrOutputParser()


@lru_cache
def _cross_encoder(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device="cuda", max_length=512)   # chunks are <=~900 chars + heading, fits 512 tokens


def rrf(rankings, k: int = 60):
    """Merge ranked lists: score = sum of 1 / (k + rank). k=60 is the standard constant; it damps the top ranks
    so agreement between lists beats a single first place."""
    scores, docs = {}, {}
    for ranking in rankings:
        for r, d in enumerate(ranking, 1):
            key = (d.metadata["source"], d.page_content)
            scores[key] = scores.get(key, 0) + 1 / (k + r)
            docs[key] = d
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def bm25_index(chunks):
    """Keyword search over the same chunks as the vector store. Returns search(query, k) -> docs.
    ponytail: in-memory, rebuilt per process (~1 s for 11k chunks). Qdrant sparse vectors if it moves to server mode (M5)."""
    idx = BM25Okapi([tokens(d.page_content) for d in chunks])
    return lambda q, k: [chunks[i] for i in np.argsort(idx.get_scores(tokens(q)))[::-1][:k]]


def search(db, question: str, pool: int = 20, rewrite: str | None = None, bm25=None):
    """`rewrite` = precomputed rewritten query (batched by the caller), or None. `bm25` = bm25_index(...) for hybrid."""
    rankings = [db.similarity_search(question, k=pool)]
    if rewrite:
        rankings.append(db.similarity_search(rewrite, k=pool))
    if bm25:
        rankings.append(bm25(question, pool))
    return corpus.unique(rrf(rankings))[:pool]


JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_ASK = "contain the actual answer to the user's question (the fact, value, code or steps), not just the same topic"


def _jev(state: str, questions: dict) -> dict:
    """TypeSafe Jev: typed answers with probabilities instead of text. Retries 429/529 as the API asks."""
    for attempt in range(4):
        r = httpx.post(JEV_URL, headers={"Authorization": f"Bearer {os.environ['JEV_API']}"}, timeout=30,
                       json={"model": "jev-latest", "state": state, "questions": questions})
        if r.status_code not in (429, 529):
            break
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r.json()["answers"]


def _jev_scores(question: str, docs, style: str) -> list[float]:
    """noul: one yes/no per passage (pointwise), all in one request. choice: one pick over all passages (listwise distribution)."""
    state = f"A user asked the LangChain documentation: {question}"
    if style == "noul":
        ans = _jev(state, {f"p{i}": {"type": "noul", "instructions": f"Does this passage {JEV_ASK}?\n\nPassage:\n{d.page_content}"}
                           for i, d in enumerate(docs)})
        return [ans[f"p{i}"]["noul"] for i in range(len(docs))]
    ans = _jev(state, {"c": {"type": "choice", "instructions": f"Which passage does {JEV_ASK}?",
                             "criteria": {f"p{i}": d.page_content for i, d in enumerate(docs)}}})
    probs = ans["c"]["probabilities"]
    return [probs.get(f"p{i}", 0.0) for i in range(len(docs))]


def rerank(question: str, docs, top: int = 5, model: str | None = None):
    """model: None = RERANK_MODEL from .env, else the generator LLM; "ce:<hf id>" = local cross-encoder (scores every pair, reorders all);
    "jev:noul" / "jev:choice" = TypeSafe Jev probabilities (pointwise / listwise);
    any other name = that LLM via the proxy (listwise: top `top` in its order, then the rest in original order)."""
    model = model or os.environ.get("RERANK_MODEL")   # default set in .env; unset -> generator LLM
    if model in ("jev:noul", "jev:choice"):
        scores = _jev_scores(question, docs, model[4:])
        return [docs[i] for i in sorted(range(len(docs)), key=lambda i: -scores[i])]   # stable: ties (2-decimal probs) keep dense order
    if model and model.startswith("ce:"):
        scores = _cross_encoder(model[3:]).predict([(question, d.page_content) for d in docs])
        return [docs[i] for i in np.argsort(-scores)]
    chain = _llm_reranker(model) if model else rerank_chain
    passages = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs))
    reply = chain.invoke({"question": question, "passages": passages, "top": top})
    picked = list(dict.fromkeys(i for i in map(int, re.findall(r"\d+", reply)) if i < len(docs)))[:top]
    return [docs[i] for i in picked] + [d for i, d in enumerate(docs) if i not in picked]


if __name__ == "__main__":
    from langchain_core.documents import Document
    d = lambda t: Document(page_content=t, metadata={"source": "s"})
    a, b, c = d("a"), d("b"), d("c")
    # b is 2nd in both lists, a and c are 1st in only one: agreement wins
    assert [x.page_content for x in rrf([[a, b], [c, b]])] == ["b", "a", "c"]
    print("ok")
