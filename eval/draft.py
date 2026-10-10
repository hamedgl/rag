"""Draft eval items with a strong LLM, for human review.

1. Sample chunks stratified by product area, give each a question type.
2. LLM writes a user-style question (no copying the doc wording), answer and verbatim evidence quotes. Non-verbatim quotes are dropped.
3. Pooling: retrieve top-20 for each question; the LLM marks every chunk that also answers it and quotes it, giving extra evidence.
4. "Should decline" items: LLM proposes a plausible follow-up the chunk does NOT answer; kept only if pooling finds no answer in the top 20.
Writes eval/drafts.jsonl + eval/review.md.
"""
import json, random, re
from collections import Counter
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag import corpus, generate, store

EVAL = Path(__file__).resolve().parent
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
llm = generate.judge_llm   # the strong, fixed model, not the generator under test

AREAS = {"langchain": 15, "langgraph": 10, "deepagents": 10, "python": 8, "other": 3}   # ponytail: hand-set mix (47 + declines)
TYPES = ["lookup", "how-to", "symptom (user describes a problem, not the feature name)", "exact value or default", "code (answer is an import, call or parameter)"]

GEN = ChatPromptTemplate.from_messages([("human", """You write evaluation questions for a documentation search system.

Documentation chunk (section: {section}):
<<<
{text}
>>>

Write ONE question of type: {qtype}
Rules:
- A real user would ask it WITHOUT having seen this chunk: use your own words, do not copy distinctive phrases from the chunk.
- The chunk alone must fully answer it.
- evidence: 1-3 short quotes (max 150 chars each) copied EXACTLY, character for character, from the chunk, that prove the answer.
If the chunk is too thin for a good question (only links, only a heading, boilerplate), return {{"skip": true}}.

Return only JSON: {{"q": "...", "expected": "...", "evidence": ["..."]}}""")]) | llm | StrOutputParser()

DECLINE = ChatPromptTemplate.from_messages([("human", """Documentation chunk (section: {section}):
<<<
{text}
>>>
Write ONE plausible follow-up question a user might ask about this topic that this chunk does NOT answer and that docs would typically not cover
(e.g. an exact internal default, a benchmark number, a pricing detail, a version-specific edge case).
Return only JSON: {{"q": "..."}}""")]) | llm | StrOutputParser()

POOL = ChatPromptTemplate.from_messages([("human", """Question: {q}

Candidate chunks:
{chunks}

Which candidates contain information that answers the question (fully or the key part)? For each, copy ONE short exact quote (max 150 chars) from it that shows this.
Return only JSON: {{"relevant": [{{"id": <number>, "quote": "..."}}]}}  (empty list if none)""")]) | llm | StrOutputParser()


def parse(s):
    m = re.search(r"\{.*\}", s, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def area(source):
    top = source.split("/")[0]
    return top if top in AREAS else "other"


def main(seed=42, declines=6):
    rnd = random.Random(seed)
    chunks = [c for c in corpus.build() if len(c.page_content) > 400]
    by_area = {a: [c for c in chunks if area(c.metadata["source"]) == a] for a in AREAS}
    picks = [(c, TYPES[i % len(TYPES)]) for a, n in AREAS.items() for i, c in enumerate(rnd.sample(by_area[a], n + 4))]  # +4 spare per area for skips

    raw = GEN.batch([{"section": c.metadata["section"], "text": c.page_content, "qtype": t} for c, t in picks], config={"max_concurrency": 8})
    items, need = [], dict(AREAS)
    for (c, t), r in zip(picks, raw):
        d, a = parse(r), area(c.metadata["source"])
        if not d or d.get("skip") or need[a] == 0:
            continue
        ev = [e for e in d.get("evidence", []) if norm(e) and norm(e) in norm(c.page_content)]   # verbatim or dropped
        if not ev:
            continue
        need[a] -= 1
        items.append(dict(type=t.split(" ")[0], q=d["q"], expected=d["expected"], evidence=ev,
                          seed_source=c.metadata["source"], seed_section=c.metadata["section"]))

    for c in rnd.sample(chunks, declines * 2):
        d = parse(DECLINE.invoke({"section": c.metadata["section"], "text": c.page_content}))
        if d and d.get("q"):
            items.append(dict(type="decline", q=d["q"], expected="Not in docs", evidence=[],
                              seed_source=c.metadata["source"], seed_section=c.metadata["section"]))

    db = store.get_store("full-v3", corpus.build)
    pools = [db.similarity_search(it["q"], k=20) for it in items]
    judged = POOL.batch([{"q": it["q"], "chunks": "\n\n".join(f"[{i}] {d.page_content[:1200]}" for i, d in enumerate(p))}
                         for it, p in zip(items, pools)], config={"max_concurrency": 8})
    kept = []
    for it, p, r in zip(items, pools, judged):
        rel = (parse(r) or {}).get("relevant", [])
        rel = rel if isinstance(rel, list) else []
        quotes = [x["quote"] for x in rel if isinstance(x, dict) and isinstance(x.get("id"), int) and 0 <= x["id"] < len(p)
                  and norm(x.get("quote", "")) and norm(x["quote"]) in norm(p[x["id"]].page_content)]
        if it["type"] == "decline":
            if not quotes and sum(i["type"] == "decline" for i in kept) < declines:
                kept.append(it)
            continue
        it["evidence"] = list(dict.fromkeys(it["evidence"] + quotes))
        it["overlap"] = round(len(set(norm(it["q"]).split()) & set(norm(" ".join(it["evidence"])).split())) / max(1, len(set(norm(it["q"]).split()))), 2)
        kept.append(it)

    for i, it in enumerate(kept, 1):
        it["id"] = f"d{i}"
    (EVAL / "drafts.jsonl").write_text("".join(json.dumps(it) + "\n" for it in kept))

    lines = ["# Eval drafts: review\n", "Reply with e.g. `drop d3 d17`, `fix d22: <better question or answer>`, or `all good`.\n",
             f"{len(kept)} items: {dict(Counter(i['type'] for i in kept))}\n"]
    for it in kept:
        lines += [f"\n## {it['id']} · {it['type']} · {it['seed_source']}", f"**Q:** {it['q']}", f"**Expected:** {it['expected']}",
                  f"**Evidence ({len(it['evidence'])}):** " + " · ".join(f"`{e[:90]}`" for e in it["evidence"][:4])]
    (EVAL / "review.md").write_text("\n".join(lines) + "\n")
    print(f"{len(kept)} drafts → eval/review.md", dict(Counter(i["type"] for i in kept)))


if __name__ == "__main__":
    main()
