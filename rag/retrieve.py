"""Retrieval pipeline: dense search (+ optional rewrite fused by RRF), then listwise LLM rerank."""
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import corpus, generate

rerank_chain = ChatPromptTemplate.from_messages([("human", """Rank documentation passages by how useful they are for answering the question.
A useful passage contains the actual answer (the fact, value, code or steps), not just the same topic.

Question: {question}

Passages:
{passages}

Reply with the numbers of the {top} most useful passages, best first, comma-separated (e.g. 3, 0, 7). Numbers only.""")]) | generate.llm | StrOutputParser()


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


def search(db, question: str, pool: int = 20, rewrite: str | None = None):
    """`rewrite` = precomputed rewritten query (batched by the caller), or None for the plain question."""
    rankings = [db.similarity_search(question, k=pool)]
    if rewrite:
        rankings.append(db.similarity_search(rewrite, k=pool))
    return corpus.unique(rrf(rankings))[:pool]


def rerank(question: str, docs, top: int = 5):
    """Top `top` in the LLM's order, then the rest in their original order. Unparseable reply -> original order."""
    passages = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs))
    reply = rerank_chain.invoke({"question": question, "passages": passages, "top": top})
    picked = list(dict.fromkeys(i for i in map(int, re.findall(r"\d+", reply)) if i < len(docs)))[:top]
    return [docs[i] for i in picked] + [d for i, d in enumerate(docs) if i not in picked]


if __name__ == "__main__":
    from langchain_core.documents import Document
    d = lambda t: Document(page_content=t, metadata={"source": "s"})
    a, b, c = d("a"), d("b"), d("c")
    # b is 2nd in both lists, a and c are 1st in only one: agreement wins
    assert [x.page_content for x in rrf([[a, b], [c, b]])] == ["b", "a", "c"]
    print("ok")
