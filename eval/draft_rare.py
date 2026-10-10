"""Draft rare-token eval items (error strings, env vars / config keys, imports): the cases where keyword search should win.

Same flow as draft.py, two differences:
- The question MUST contain the exact rare token, the way a user pastes it from a terminal or code.
- Pooling labels evidence over dense ∪ BM25, so the labels don't favour the method under test.
Writes eval/rare_drafts.jsonl + eval/review_rare.md.
"""
import json, random, re
from collections import Counter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from eval.draft import EVAL, POOL, llm, norm, parse
from rag import corpus, retrieve, store

KINDS = {   # kind -> (regex for the rare string, how many to draft)
    "error": (r"\b\w+(?:Error|Exception)\b:? [^\n`]{15,120}", 5),
    "env/config": (r"\b[A-Z][A-Z0-9]+_[A-Z0-9_]{3,}\b", 5),
    "import": (r"^(?:from [\w.]+ import [\w, ]+|import [\w.]+)$", 5),
}

GEN = ChatPromptTemplate.from_messages([("human", """You write evaluation questions for a documentation search system.

Documentation chunk (section: {section}):
<<<
{text}
>>>

Write ONE question a real user would ask that contains this exact string, copied character for character: {token}
(as if they pasted it from their terminal, config or code). Apart from that string, use your own words.
The chunk alone must fully answer it.
evidence: 1-3 short quotes (max 150 chars each) copied EXACTLY from the chunk that prove the answer.
If the chunk can't support a good question, return {{"skip": true}}.

Return only JSON: {{"q": "...", "expected": "...", "evidence": ["..."]}}""")]) | llm | StrOutputParser()


def main(seed=7):
    rnd = random.Random(seed)
    chunks = corpus.build()
    df = Counter(t for c in chunks for t in set(retrieve.tokens(c.page_content)))
    rare = lambda s: min(df[t] for t in retrieve.tokens(s)) <= 5   # the string's rarest token appears in at most 5 chunks
    picks = []
    for kind, (pat, n) in KINDS.items():
        cands = [(c, m.group(0).strip()) for c in chunks for m in re.finditer(pat, corpus.body(c), re.M) if rare(m.group(0))]
        rnd.shuffle(cands)
        seen = set()   # one pick per page, for spread
        for c, tok in cands:
            if c.metadata["source"] not in seen and len(seen) < n * 2:   # 2x spare for skips
                seen.add(c.metadata["source"])
                picks.append((kind, c, tok))

    raw = GEN.batch([{"section": c.metadata["section"], "text": c.page_content, "token": t} for _, c, t in picks], config={"max_concurrency": 8})
    items, need = [], {k: n for k, (_, n) in KINDS.items()}
    for (kind, c, tok), r in zip(picks, raw):
        d = parse(r)
        if not d or d.get("skip") or need[kind] == 0 or tok not in d.get("q", ""):
            continue
        ev = [e for e in d.get("evidence", []) if norm(e) and norm(e) in norm(c.page_content)]
        if ev:
            need[kind] -= 1
            items.append(dict(type=kind, q=d["q"], expected=d["expected"], evidence=ev, token=tok,
                              seed_source=c.metadata["source"], seed_section=c.metadata["section"]))

    db, bm = store.get_store("full-v3", chunks), retrieve.bm25_index(chunks)
    pools = [corpus.unique(retrieve.rrf([db.similarity_search(it["q"], k=20), bm(it["q"], 20)])) for it in items]   # union, ~30-40
    judged = POOL.batch([{"q": it["q"], "chunks": "\n\n".join(f"[{i}] {d.page_content[:1200]}" for i, d in enumerate(p))}
                         for it, p in zip(items, pools)], config={"max_concurrency": 8})
    for it, p, r in zip(items, pools, judged):
        rel = (parse(r) or {}).get("relevant", [])
        quotes = [x["quote"] for x in (rel if isinstance(rel, list) else []) if isinstance(x, dict) and isinstance(x.get("id"), int)
                  and 0 <= x["id"] < len(p) and norm(x.get("quote", "")) and norm(x["quote"]) in norm(p[x["id"]].page_content)]
        it["evidence"] = list(dict.fromkeys(it["evidence"] + quotes))

    for i, it in enumerate(items, 1):
        it["id"] = f"r{i}"
    (EVAL / "rare_drafts.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
    lines = ["# Rare-token drafts: review\n", "Reply with e.g. `drop r3`, `fix r2: <better question>`, or `all good`.\n"]
    for it in items:
        lines += [f"\n## {it['id']} · {it['type']} · {it['seed_source']}", f"**Token:** `{it['token']}`", f"**Q:** {it['q']}",
                  f"**Expected:** {it['expected']}", f"**Evidence ({len(it['evidence'])}):** " + " · ".join(f"`{e[:90]}`" for e in it["evidence"][:3])]
    (EVAL / "review_rare.md").write_text("\n".join(lines) + "\n")
    print(f"{len(items)} drafts → eval/review_rare.md", dict(Counter(i["type"] for i in items)))


if __name__ == "__main__":
    main()
