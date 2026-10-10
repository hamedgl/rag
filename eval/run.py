"""Run an eval set against a corpus; chunk-level retrieval metrics + judged answers; fail on regression.

Relevance is evidence-based: a retrieved chunk is relevant if it contains one of the item's evidence quotes,
so labels survive re-chunking. Items with no evidence are "should decline" questions (retrieval not scored).
"""
import argparse, json, re, sys, time
from pathlib import Path

from rag import corpus, generate, store

EVAL = Path(__file__).resolve().parent
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()


def load_items(sets):
    return [json.loads(l) for s in sets for l in (EVAL / f"{s}.jsonl").read_text().splitlines() if l.strip()]


def run(items, db, k=4, pool=20):
    rows = []
    for it in items:
        hits = db.similarity_search(it["q"], k=pool)
        ev = [norm(e) for e in it["evidence"]]
        rel = [i + 1 for i, d in enumerate(hits) if any(e in norm(d.page_content) for e in ev)]
        ans, model = generate.answer(it["q"], hits[:k])
        rows.append({**it, "rank": rel[0] if rel else None, "got": [d.metadata["source"] for d in hits[:k]],
                     "answer": ans, "model": model, "ok": generate.judge(it["q"], it["expected"], ans)})
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
    ap.add_argument("--gate", action="store_true", help="exit 1 if below eval/thresholds.json")
    a = ap.parse_args()

    t = time.time()
    db = store.get_store(f"{a.corpus}-v3", lambda: corpus.build(corpus.SUBSET if a.corpus == "subset" else None))
    print(f"[{a.corpus}] {db._collection.count()} chunks ready in {time.time() - t:.0f}s")

    rows = run(load_items(a.sets), db, k=a.k)
    m = metrics(rows, a.k)
    for r in rows:
        print(f"  {'✓' if r['ok'] else '✗'} {r['id']:<5} rank={str(r['rank']):<5} {r['type']:<20} {r['q'][:70]}")
    models = {r["model"] for r in rows}
    print(json.dumps(m), "| answered by:", models, "| judge:", generate.judge_llm.model_name)
    if len(models) > 1:
        print("WARNING: mixed generator models (gateway fallback?) - results are not from one model")

    out = EVAL / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.corpus}-k{a.k}.json"
    out.write_text(json.dumps({"args": vars(a), "metrics": m, "rows": rows}, indent=1))

    if a.gate:
        th = json.loads((EVAL / "thresholds.json").read_text())
        bad = {key: (m[key], v) for key, v in th.items() if m.get(key, 0) < v}
        if bad:
            sys.exit(f"GATE FAILED: {bad}")
        print("gate passed")


if __name__ == "__main__":
    main()
