"""Retrieval pipeline: dense search (+ optional rewrite fused by RRF), then listwise LLM rerank."""
import os
import re
import time
from functools import lru_cache

import httpx
import numpy as np
from qdrant_client import models
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


def version_filter(version: str | None):
    """Qdrant filter for one docs version. v1 = "not v0.3": chunks indexed before versions existed have no field."""
    if version is None:
        return None
    cond = [models.FieldCondition(key="metadata.version", match=models.MatchValue(value="v0.3"))]
    return models.Filter(must=cond) if version == "v0.3" else models.Filter(must_not=cond)


def search(db, question: str, pool: int = 20, rewrite: str | None = None, bm25=None, version: str | None = None):
    """`rewrite` = precomputed rewritten query (batched by the caller), or None. `bm25` = bm25_index(...) for hybrid.
    `version` = search only that docs version (filtered in Qdrant, so the pool fills with the right version)."""
    f = version_filter(version)
    rankings = [db.similarity_search(question, k=pool, filter=f)]
    if rewrite:
        rankings.append(db.similarity_search(rewrite, k=pool, filter=f))
    if bm25:
        in_version = lambda d: version is None or (d.metadata.get("version", "v1") == "v0.3") == (version == "v0.3")
        rankings.append([d for d in bm25(question, pool * 2) if in_version(d)][:pool])
    return corpus.unique(rrf(rankings))[:pool]


# Laya = open-weights Jev clone serving the same API (local: `uvx --from "laya[serve]" laya-serve`)
APIS = {"jev": ("https://api.typesafe.ai/v1/systemone", "JEV_API"),
        "laya": (os.environ.get("LAYA_URL", "http://127.0.0.1:8000/v1/systemone"), "LAYA_API_KEY")}
JEV_ASK = "contain the actual answer to the user's question (the fact, value, code or steps), not just the same topic"


def _jev(state: str, questions: dict, api: str = "jev") -> dict:
    """TypeSafe Jev: typed answers with probabilities instead of text. Retries 429/529 as the API asks."""
    url, key = APIS[api]
    for attempt in range(4):
        r = httpx.post(url, headers={"Authorization": f"Bearer {os.environ.get(key, 'none')}"}, timeout=30,
                       json={"model": "jev-latest", "state": state, "questions": questions})
        if r.status_code not in (429, 529):
            break
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r.json()["answers"]


def _jev_scores(question: str, docs, style: str, api: str = "jev", checked: int = 4) -> list[float]:
    """noul: one yes/no per passage (pointwise), all in one request. choice: one pick over all passages (listwise distribution)."""
    state = f"A user asked the LangChain documentation: {question}"
    if style == "noul":
        ans = _jev(state, {f"p{i}": {"type": "noul", "instructions": f"Does this passage {JEV_ASK}?\n\nPassage:\n{d.page_content}"}
                           for i, d in enumerate(docs)}, api)
        return [ans[f"p{i}"]["noul"] for i in range(len(docs))]
    # choice ranks, but its probabilities sum to 1 over the pool, so it can't say "none of these answer". A second
    # request asks one noul per passage of the new top `checked`: an absolute P(answers), read by confident() for the
    # version fallback. Was one request with 20 nouls: same latency, but ~2x the tokens, 16 of the nouls never read.
    probs = _jev(state, {"c": {"type": "choice", "instructions": f"Which passage does {JEV_ASK}?",
                               "criteria": {f"p{i}": d.page_content for i, d in enumerate(docs)}}}, api)["c"]["probabilities"]
    scores = [probs.get(f"p{i}", 0.0) for i in range(len(docs))]
    top = sorted(range(len(docs)), key=lambda i: -scores[i])[:checked]   # same stable order rerank() applies
    ans = _jev(state, {f"a{i}": {"type": "noul", "instructions": f"Does this passage {JEV_ASK}?\n\nPassage:\n{docs[i].page_content}"}
                       for i in top}, api)
    for i in top:
        docs[i].metadata["answers_p"] = ans[f"a{i}"]["noul"]
    return scores


def rerank(question: str, docs, top: int = 5, model: str | None = None):
    """model: None = RERANK_MODEL from .env, else the generator LLM; "ce:<hf id>" = local cross-encoder (scores every pair, reorders all);
    "jev:noul" / "jev:choice" = TypeSafe Jev probabilities (pointwise / listwise); "laya:noul" / "laya:choice" = same on Laya;
    any other name = that LLM via the proxy (listwise: top `top` in its order, then the rest in original order)."""
    model = model or os.environ.get("RERANK_MODEL")   # default set in .env; unset -> generator LLM
    api, _, style = (model or "").partition(":")
    if api in APIS and style in ("noul", "choice"):
        scores = _jev_scores(question, docs, style, api)
        for d, sc in zip(docs, scores):
            d.metadata["rerank_score"] = sc
        return [docs[i] for i in sorted(range(len(docs)), key=lambda i: -scores[i])]   # stable: ties (2-decimal probs) keep dense order
    if model and model.startswith("ce:"):
        scores = _cross_encoder(model[3:]).predict([(question, d.page_content) for d in docs])
        return [docs[i] for i in np.argsort(-scores)]
    chain = _llm_reranker(model) if model else rerank_chain
    passages = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs))
    reply = chain.invoke({"question": question, "passages": passages, "top": top})
    picked = list(dict.fromkeys(i for i in map(int, re.findall(r"\d+", reply)) if i < len(docs)))[:top]
    return [docs[i] for i in picked] + [d for i, d in enumerate(docs) if i not in picked]


VERSIONS = {
    "v1": "current LangChain v1, LangGraph, Deep Agents or LangSmith, or no particular version mentioned",
    "v0.3": "legacy LangChain v0.x: mentions v0.3, v0.2 or an older version, or legacy APIs such as AgentExecutor, "
            "initialize_agent, LLMChain, ConversationBufferMemory or RunnableWithMessageHistory",
}


def route_version(question: str, threshold: float = 0.7) -> str:
    """Which docs version the question is about. Latest unless the router is confident it's legacy."""
    ans = _jev(f"A user asked the LangChain documentation: {question}",
               {"v": {"type": "choice", "instructions": "Which version of the documentation is the user asking about?", "criteria": VERSIONS}})
    return "v0.3" if ans["v"]["probabilities"].get("v0.3", 0) >= threshold else "v1"



def confident(docs, k: int = 4, threshold: float = 0.5) -> bool:
    """Does any of the top k reranked passages answer the question (absolute noul P(yes), not choice's relative share)?
    Unscored pools (non-Jev rerankers) count as confident: no fallback."""
    return max((d.metadata.get("answers_p", 1.0) for d in docs[:k]), default=0) >= threshold


if __name__ == "__main__":
    from langchain_core.documents import Document
    d = lambda t: Document(page_content=t, metadata={"source": "s"})
    a, b, c = d("a"), d("b"), d("c")
    # b is 2nd in both lists, a and c are 1st in only one: agreement wins
    assert [x.page_content for x in rrf([[a, b], [c, b]])] == ["b", "a", "c"]
    print("ok")
