"""Naive (pypdf) vs layout-aware (pymupdf4llm) PDF parsing, end to end on the paper questions (eval/pdf.jsonl).
Each parser's output is chunked the same way and indexed alone in a throwaway collection; then the normal pipeline
(search -> Jev rerank -> answer -> judge) runs. Run: uv run python -m eval.pdf_parsers
"""
import re
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document
from pypdf import PdfReader

from eval.run import load_items, metrics, run
from rag import corpus, store

PDFS = Path(__file__).resolve().parent.parent / "data/pdfs"
PAPERS = {"2210.03629": "ReAct: Synergizing Reasoning and Acting in Language Models",
          "2312.10997": "Retrieval-Augmented Generation for Large Language Models: A Survey"}
PARSERS = {"naive": lambda f: "\n".join(p.extract_text() or "" for p in PdfReader(f).pages),
           "layout": lambda f: pymupdf4llm.to_markdown(str(f))}


def main():
    items = load_items(["pdf"])
    for name, parse in PARSERS.items():
        docs = [Document(page_content=parse(PDFS / f"{pid}.pdf"), metadata={"source": f"papers/{pid}", "title": title, "version": "v1"})
                for pid, title in PAPERS.items()]
        chunks = corpus.chunk(docs)
        db = store.get_store(f"pdf-{name}", chunks)
        rows = run(items, db, rerank=True)
        print(f"{name:7} {len(chunks)} chunks | {metrics(rows, 4)}")
        for r in rows:
            print(f"   {'✓' if r['ok'] else '✗'} {r['id']} rank={r['rank']} {r['answer'][:110]!r}")
        store.client().delete_collection(f"pdf-{name}")


if __name__ == "__main__":
    main()
