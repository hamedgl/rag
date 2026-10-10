"""Latency and token cost per chat turn, against the running service (make up).
Latency: /chat timings for the hard set, sequential (one user). Tokens: the same answer prompts replayed non-streamed,
because the proxy drops usage on streamed responses; the proxy's hidden prompt is measured with a near-empty call.
Run: uv run python -m eval.cost_report"""
import json, statistics as st

import httpx

from eval.run import load_items
from rag import corpus, generate, retrieve, store

API = "http://localhost:8000/chat"
STAGES = ["condense", "retrieve", "rerank", "first_token", "total"]


def chat(q):
    with httpx.stream("POST", API, json={"messages": [{"role": "user", "content": q}]}, timeout=120) as r:
        for line in r.iter_lines():
            if line.startswith('data: {"id"'):
                return json.loads(line[6:])["timings"]


def main():
    qs = [it["q"] for it in load_items(["hard"])]
    chat(qs[0])   # warm-up
    ts = [chat(q) for q in qs]
    cum = {s: [t[s] for t in ts if s in t] for s in STAGES}
    prev, print_rows = 0, []
    for s in STAGES:   # timings are cumulative: report each stage's own share
        med = st.median(cum[s])
        print_rows.append((s, med - prev if s != "total" else med, max(cum[s])))
        prev = med if s != "total" else prev
    print(f"latency, {len(qs)} questions sequential (median stage time / max cumulative):")
    for s, own, mx in print_rows:
        print(f"  {s:12} {own:5.2f}s   (max {mx:.2f}s)")

    hidden = generate.llm.invoke("Hi").usage_metadata["input_tokens"]
    db = store.get_store("full-v3", corpus.build())
    use = []
    for q in qs:
        docs = retrieve.rerank(q, retrieve.search(db, q))[:4]
        msg = generate.answer_chain.invoke({"context": generate.format_context(docs), "question": q})
        use.append(msg.usage_metadata)
    inp, out = st.median(u["input_tokens"] for u in use), st.median(u["output_tokens"] for u in use)
    print(f"\nanswer call tokens (median): input {inp:.0f} (of which ~{hidden} hidden proxy prompt = {hidden / inp:.0%}), output {out:.0f}")
    print(f"jev rerank: ~{st.median(len(' '.join(d.page_content for d in retrieve.search(db, q))) // 4 for q in qs):.0f} input tokens/query "
          f"(chars/4 estimate) at $0.042/M = ${0.042e-6 * st.median(len(' '.join(d.page_content for d in retrieve.search(db, q))) // 4 for q in qs):.6f}")


if __name__ == "__main__":
    main()
