"""Run an eval set against a corpus; chunk-level retrieval metrics + judged answers; fail on regression.

Relevance is evidence-based: a retrieved chunk is relevant if it contains one of the item's evidence quotes,
so labels survive re-chunking. Items with no evidence are "should decline" questions (retrieval not scored).
"""
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag import corpus, generate, retrieve, store

EVAL = Path(__file__).resolve().parent
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()


def load_items(sets):
    return [json.loads(l) for s in sets for l in (EVAL / f"{s}.jsonl").read_text().splitlines() if l.strip()]


def pmap(f, xs, workers=8):
    """LLM calls are network-bound: run them in threads. ponytail: fixed 8, lower it if the proxy starts rate-limiting."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(f, xs))


def run(items, db, k=4, pool=20, rewrite=False, rerank=False):
    qs = [it["q"] for it in items]
    queries = pmap(generate.rewrite, qs) if rewrite else [None] * len(qs)   # retrieval only; the answer sees the original question
    pools = [retrieve.search(db, q, pool, rw) for q, rw in zip(qs, queries)]   # sequential: embedded Qdrant is single-process, ms per query
    if rerank:
        pools = pmap(lambda x: retrieve.rerank(*x), list(zip(qs, pools)))
    answers = pmap(lambda x: generate.answer(*x), [(q, h[:k]) for q, h in zip(qs, pools)])
    oks = pmap(lambda x: generate.judge(*x), [(it["q"], it["expected"], a) for it, (a, _) in zip(items, answers)])
    rows = []
    for it, query, hits, (ans, model), ok in zip(items, queries, pools, answers, oks):
        ev = [norm(e) for e in it["evidence"]]
        rel = [i + 1 for i, d in enumerate(hits) if any(e in norm(d.page_content) for e in ev)]
        rows.append({**it, "query": query, "rank": rel[0] if rel else None, "got": [d.metadata["source"] for d in hits[:k]],
                     "answer": ans, "model": model, "ok": ok})
    return rows


def metrics(rows, k):
    scored = [r for r in rows if r["evidence"]]
    return {
        f"hit@{k}": round(sum(bool(r["rank"]) and r["rank"] <= k for r in scored) / len(scored), 3),
        "mrr@20": round(sum(1 / r["rank"] for r in scored if r["rank"]) / len(scored), 3),
        "answer_acc": round(sum(r["ok"] for r in rows) / len(rows), 3),
        "decline_acc": round(sum(r["ok"] for r in rows if not r["evidence"]) / max(1, sum(not r["evidence"] for r in rows)), 3),
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["full", "subset"], default="full")
    ap.add_argument("--sets", nargs="+", default=["hard"])
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--rewrite", action="store_true", help="LLM-rewrite the question before retrieval")
    ap.add_argument("--rerank", action="store_true", help="listwise LLM rerank of the top 20")
    ap.add_argument("--gate", action="store_true", help="exit 1 if below eval/thresholds.json")
    a = ap.parse_args()

    t = time.time()
    db = store.get_store(f"{a.corpus}-v3", corpus.build(corpus.SUBSET if a.corpus == "subset" else None))
    print(f"[{a.corpus}] ready in {time.time() - t:.0f}s")

    rows = run(load_items(a.sets), db, k=a.k, rewrite=a.rewrite, rerank=a.rerank)
    m = metrics(rows, a.k)
    for r in rows:
        print(f"  {'✓' if r['ok'] else '✗'} {r['id']:<5} rank={str(r['rank']):<5} {r['type']:<20} {r['q'][:70]}")
    models = {r["model"] for r in rows}
    print(json.dumps(m), "| answered by:", models, "| judge:", generate.judge_llm.model_name)
    if len(models) > 1:
        print("WARNING: mixed generator models (gateway fallback?) - results are not from one model")

    out = EVAL / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.corpus}-k{a.k}{'-rw' if a.rewrite else ''}{'-rr' if a.rerank else ''}.json"
    out.write_text(json.dumps({"args": vars(a), "metrics": m, "rows": rows}, indent=1))

    if a.gate:
        th = json.loads((EVAL / "thresholds.json").read_text())
        bad = {key: (m[key], v) for key, v in th.items() if m.get(key, 0) < v}
        if bad:
            sys.exit(f"GATE FAILED: {bad}")
        print("gate passed")


if __name__ == "__main__":
    main()
