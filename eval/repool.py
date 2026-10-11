"""Re-pool relevance labels after the corpus changes. Labels are evidence quotes; a corpus change adds chunks that also
answer a question (new pages, inlined snippets) in different words, and the metrics would count finding them as a miss.
For each item with evidence: the pinned judge reads the top 20 of dense ∪ BM25 (so labels don't favour one method) and
every chunk that answers gets a verbatim quote added. Quotes are only ADDED, never removed.
Run: uv run python -m eval.repool hard main rare multi langsmith
"""
import json, sys

from eval.draft import EVAL, POOL, norm, parse
from eval.run import pmap
from rag import corpus, retrieve, store


def main(sets):
    chunks = corpus.build()
    db, bm = store.get_store("full-v3", chunks), retrieve.bm25_index(chunks)
    for s in sets:
        path = EVAL / f"{s}.jsonl"
        items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        todo = [it for it in items if it["evidence"]]
        q = lambda it: it.get("standalone", it["q"])
        pools = [corpus.unique(retrieve.rrf([db.similarity_search(q(it), k=20), bm(q(it), 20)])) for it in todo]
        judged = pmap(lambda x: POOL.invoke({"q": q(x[0]), "chunks": "\n\n".join(f"[{i}] {d.page_content[:1200]}" for i, d in enumerate(x[1]))}),
                      list(zip(todo, pools)))
        added = 0
        for it, pool, r in zip(todo, pools, judged):
            rel = (parse(r) or {}).get("relevant", [])
            quotes = [x["quote"] for x in (rel if isinstance(rel, list) else []) if isinstance(x, dict) and isinstance(x.get("id"), int)
                      and 0 <= x["id"] < len(pool) and norm(x.get("quote", "")) and norm(x["quote"]) in norm(pool[x["id"]].page_content)]
            new = [e for e in dict.fromkeys(quotes) if norm(e) not in {norm(v) for v in it["evidence"]}]
            it["evidence"] += new
            added += len(new)
        path.write_text("".join(json.dumps(it) + "\n" for it in items))
        print(f"{s}: +{added} evidence quotes over {len(todo)} items")


if __name__ == "__main__":
    main(sys.argv[1:])
