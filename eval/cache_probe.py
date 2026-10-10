"""Is a semantic cache safe? For each eval question the judge LLM writes a paraphrase (same answer: the cache should hit)
and a hard negative (near-identical wording, different answer: it must not hit). Sweep the cosine threshold and report
hit rate vs wrong-hit rate. Writes eval/cache_pairs.jsonl (reused on re-runs). Run: uv run python -m eval.cache_probe"""
import json
from pathlib import Path

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from eval.draft import parse
from eval.run import EVAL, load_items
from rag import generate, store

PAIRS = EVAL / "cache_pairs.jsonl"
GEN = ChatPromptTemplate.from_messages([("human", """Question about the LangChain docs: {q}

Write two new questions:
1. "paraphrase": same meaning and same correct answer, worded differently (as another user would ask it).
2. "negative": as close as possible in wording, but asking for something DIFFERENT, so the correct answer differs
   (change the class, parameter, product, value or condition asked about; keep everything else).
Return only JSON: {{"paraphrase": "...", "negative": "..."}}""")]) | generate.judge_llm | StrOutputParser()


def pairs():
    if PAIRS.exists():
        return [json.loads(l) for l in PAIRS.read_text().splitlines()]
    qs = [it.get("standalone", it["q"]) for it in load_items(["hard", "main", "rare", "multi"])]
    out = [dict(q=q, **d) for q, d in zip(qs, map(parse, GEN.batch([{"q": q} for q in qs], config={"max_concurrency": 8})))
           if d and d.get("paraphrase") and d.get("negative")]
    PAIRS.write_text("".join(json.dumps(p) + "\n" for p in out))
    return out


def main():
    ps = pairs()
    emb = lambda xs: np.array([store.embedder().embed_query(x) for x in xs])   # query prompt: cache keys are queries
    Q, P, N = emb([p["q"] for p in ps]), emb([p["paraphrase"] for p in ps]), emb([p["negative"] for p in ps])
    cos = lambda a, b: np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    sp, sn = cos(Q, P), cos(Q, N)
    S = (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ (Q / np.linalg.norm(Q, axis=1, keepdims=True)).T
    cross = S[np.triu_indices(len(Q), 1)]   # two different eval questions: a hit here is also wrong
    print(f"{len(ps)} pairs | paraphrase sim median {np.median(sp):.3f} | negative sim median {np.median(sn):.3f} | max cross-question {cross.max():.3f}")
    print(f"{'threshold':>9} {'hit rate':>9} {'wrong hit (neg)':>16} {'wrong hit (cross)':>18}")
    for t in (0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98):
        print(f"{t:>9.2f} {np.mean(sp >= t):>9.2f} {np.mean(sn >= t):>16.2f} {np.mean(cross >= t):>18.4f}")
    worst = sorted(zip(sn, ps), key=lambda x: -x[0])[:5]
    print("\nmost similar hard negatives (a cache would serve the WRONG answer above these scores):")
    for s, p in worst:
        print(f"  {s:.3f}  {p['q'][:70]!r}\n         {p['negative'][:70]!r}")


if __name__ == "__main__":
    main()
