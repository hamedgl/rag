"""Run an eval set against a corpus; chunk-level retrieval metrics + judged answers; fail on regression.

Relevance is evidence-based: a retrieved chunk is relevant if it contains one of the item's evidence quotes,
so labels survive re-chunking. Items with no evidence are "should decline" questions (retrieval not scored).
"""
import argparse, json, os, re, sys, time
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


def timed(f):
    def g(*a):
        t = time.perf_counter()
        return f(*a), time.perf_counter() - t
    return g


def run(items, db, k=4, pool=20, rewrite=False, rerank=False, bm25=None, condense=True, rerank_model=None):
    # multi-turn items: condense history + follow-up into the question the pipeline sees (as the service does)
    asked = pmap(lambda it: generate.condense([tuple(t) for t in it["history"]], it["q"]) if condense and it.get("history") else it["q"], items)
    rewrites = pmap(generate.rewrite, asked) if rewrite else [None] * len(asked)   # retrieval only
    pools = [retrieve.search(db, q, pool, rw, bm25) for q, rw in zip(asked, rewrites)]   # sequential: embedded Qdrant is single-process, ms per query
    rr_secs = [None] * len(items)
    if rerank:
        f = timed(lambda q, p: retrieve.rerank(q, p, model=rerank_model))
        args = list(zip(asked, pools))
        # cross-encoder: one GPU model, call it sequentially; LLM rerankers are network-bound, run in threads
        out = [f(*x) for x in args] if (rerank_model or "").startswith("ce:") else pmap(lambda x: f(*x), args)
        pools, rr_secs = [o[0] for o in out], [round(o[1], 3) for o in out]
    answers = pmap(lambda x: generate.answer(*x), [(q, h[:k]) for q, h in zip(asked, pools)])
    # the judge sees the gold standalone question, so a bad condense can't be excused by a confused judge
    oks = pmap(lambda x: generate.judge(*x), [(it.get("standalone", it["q"]), it["expected"], a) for it, (a, _) in zip(items, answers)])
    rows = []
    for it, query, hits, (ans, model), ok, rr in zip(items, asked, pools, answers, oks, rr_secs):
        ev = [norm(e) for e in it["evidence"]]
        rel = [i + 1 for i, d in enumerate(hits) if any(e in norm(d.page_content) for e in ev)]
        rows.append({**it, "query": query, "rank": rel[0] if rel else None, "got": [d.metadata["source"] for d in hits[:k]],
                     "answer": ans, "model": model, "ok": ok, "rerank_s": rr})
    return rows


def metrics(rows, k):
    scored = [r for r in rows if r["evidence"]]
    return {
        f"hit@{k}": round(sum(bool(r["rank"]) and r["rank"] <= k for r in scored) / len(scored), 3),
        "recall@20": round(sum(bool(r["rank"]) for r in scored) / len(scored), 3),
        "mrr@20": round(sum(1 / r["rank"] for r in scored if r["rank"]) / len(scored), 3),
        "answer_acc": round(sum(r["ok"] for r in rows) / len(rows), 3),
        "decline_acc": round(sum(r["ok"] for r in rows if not r["evidence"]) / max(1, sum(not r["evidence"] for r in rows)), 3),
        "n": len(rows),
        **({"rerank_mean_s": round(sum(t) / len(t), 2), "rerank_p95_s": round(sorted(t)[int(0.95 * (len(t) - 1))], 2)}
           if (t := [r["rerank_s"] for r in rows if r.get("rerank_s") is not None]) else {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["full", "subset"], default="full")
    ap.add_argument("--sets", nargs="+", default=["hard"])
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--rewrite", action="store_true", help="LLM-rewrite the question before retrieval")
    ap.add_argument("--rerank", action="store_true", help="listwise LLM rerank of the top 20")
    ap.add_argument("--hybrid", action="store_true", help="fuse BM25 with dense search (RRF)")
    ap.add_argument("--no-condense", action="store_true", help="multi-turn items: retrieve with the raw follow-up")
    ap.add_argument("--rerank-model", help='LLM name via the proxy, or "ce:<hf id>" for a cross-encoder (default: generator LLM)')
    ap.add_argument("--gate", action="store_true", help="exit 1 if below eval/thresholds.json")
    a = ap.parse_args()

    t = time.time()
    chunks = corpus.build(corpus.SUBSET if a.corpus == "subset" else None)
    db = store.get_store(f"{a.corpus}-v3", chunks)
    print(f"[{a.corpus}] ready in {time.time() - t:.0f}s")

    a.rerank_model = a.rerank_model or (os.environ.get("RERANK_MODEL") if a.rerank else None)   # record what actually ran
    rows = run(load_items(a.sets), db, k=a.k, rewrite=a.rewrite, rerank=a.rerank, condense=not a.no_condense, rerank_model=a.rerank_model,
               bm25=retrieve.bm25_index(chunks) if a.hybrid else None)
    m = metrics(rows, a.k)
    for r in rows:
        print(f"  {'✓' if r['ok'] else '✗'} {r['id']:<5} rank={str(r['rank']):<5} {r['type']:<20} {r['q'][:70]}")
        if r.get("history"):
            print(f"{'':36}→ {r['query'][:100]}")
    models = {r["model"] for r in rows}
    print(json.dumps(m), "| answered by:", models, "| judge:", generate.judge_llm.model_name)
    if len(models) > 1:
        print("WARNING: mixed generator models (gateway fallback?) - results are not from one model")
    # gateway tripwire: the proxy injects a ~2k-token hidden prompt on every route (M6.4); its style must not leak into answers
    leaked = [r["id"] for r in rows if re.search(r"(?im)^\W*skipped:|\badd (?:it )?when\b|ponytail|\bYAGNI\b|★ Insight", r["answer"])]
    if leaked:
        print(f"WARNING: gateway prompt style leaked into answers: {leaked}")

    out = EVAL / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.corpus}-k{a.k}{'-rw' if a.rewrite else ''}{'-hy' if a.hybrid else ''}{'-rr' if a.rerank else ''}{'-' + a.rerank_model.split('/')[-1] if a.rerank_model else ''}{'-nc' if a.no_condense else ''}.json"
    out.write_text(json.dumps({"args": vars(a), "metrics": m, "rows": rows}, indent=1))

    if a.gate:
        th = json.loads((EVAL / "thresholds.json").read_text())
        bad = {key: (m[key], v) for key, v in th.items() if m.get(key, 0) < v}
        if bad:
            sys.exit(f"GATE FAILED: {bad}")
        print("gate passed")


if __name__ == "__main__":
    main()
