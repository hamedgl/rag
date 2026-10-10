"""Load, clean and chunk the LangChain docs (logic ported from the Phase 1 notebook, cleaner v2 + chunker v3)."""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCS = Path(__file__).resolve().parent.parent / "data/langchain-docs/src/oss"
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
MD_SEPARATORS = RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)

# Phase 1 subset, kept to compare small vs full corpus on the same eval set
SUBSET = [
    "langchain/overview", "langchain/philosophy", "langchain/quickstart", "langchain/component-architecture",
    "langchain/agents", "langchain/models", "langchain/messages", "langchain/tools",
    "langchain/structured-output", "langchain/short-term-memory", "langchain/long-term-memory",
    "langchain/streaming", "langchain/context-engineering", "langchain/guardrails",
    "langchain/human-in-the-loop", "langchain/knowledge-base", "langchain/retrieval",
    "langchain/middleware/overview", "langchain/middleware/built-in",
    "concepts/context", "concepts/memory", "concepts/providers-and-models", "concepts/products",
    "deepagents/retrieval", "deepagents/rag", "langgraph/overview", "langgraph/agentic-rag",
]


def load(path: Path) -> Document:
    raw = path.read_text()
    fm = FRONTMATTER.match(raw)
    title = re.search(r"^title:\s*(.+)$", fm.group(1), re.M) if fm else None
    rel = str(path.relative_to(DOCS).with_suffix(""))
    return Document(page_content=raw[fm.end():] if fm else raw,
                    metadata={"source": rel, "title": title.group(1).strip('"') if title else rel})


def pages(subset: list[str] | None = None) -> list[Document]:
    """All pages except the JS duplicates, or only `subset`."""
    if subset:
        paths = [DOCS / f"{s}.mdx" for s in subset]
    else:
        paths = sorted(p for p in DOCS.rglob("*.mdx") if "javascript" not in p.relative_to(DOCS).parts)
    return [load(p) for p in paths]


def _paramfield(m: re.Match) -> str:
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0)))
    name = attrs.get("path") or attrs.get("body") or attrs.get("query") or "?"
    extra = ", ".join(f"{k}: {attrs[k]}" for k in ("type", "default") if k in attrs)
    return f"Parameter `{name}`" + (f" ({extra})" if extra else "") + ":"


HTML_TAG = r"</?(?:a|div|span|br|p|ul|ol|li|table|thead|tbody|tr|td|th|code|img)\b[^>]*>"


def _prose(text: str) -> str:
    """MDX/HTML cleanup for text outside code fences (in code, `import x` and `<Foo>` are real code)."""
    text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.S)                                 # JSX comments: not rendered
    text = re.sub(r"<ParamField\s[^>]*>", _paramfield, text)                              # keep name + default before tags go
    text = re.sub(r'<[A-Z][A-Za-z]*\s[^>]*?\btitle="([^"]*)"[^>]*>', r"\n**\1**\n", text) # Tab/Accordion/Card/Step titles
    text = re.sub(r'<img\s[^>]*?\balt="([^"]*)"[^>]*>', r"\1", text)                      # images -> alt text
    text = re.sub(r"^import .*$\n?", "", text, flags=re.M)                                # snippet imports
    text = re.sub(r"</?[A-Z][A-Za-z]*(?:\s[^>]*)?/?>", "", text)                          # JSX tags, keep inner text
    text = re.sub(HTML_TAG, "", text)                                                     # HTML tags, keep inner text
    return re.sub(r"@\[(`?[^\]]+`?)\]", r"\1", text)                                     # @[`X`] -> `X`


def clean(text: str) -> str:
    text = re.sub(r"^[ \t]*:::js\n.*?^[ \t]*:::[ \t]*\n?", "", text, flags=re.S | re.M)   # JS duplicate content
    text = re.sub(r"^[ \t]*:::(?:python)?[ \t]*\n?", "", text, flags=re.M)                # unwrap Python blocks
    parts = re.split(r"(```.*?```)", text, flags=re.S)                                    # odd parts are code fences
    text = "".join(p if i % 2 else _prose(p) for i, p in enumerate(parts))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk(docs: list[Document], size: int = 800, overlap: int = 100) -> list[Document]:
    """Header-aware split, then size split; every chunk is prefixed with its heading path."""
    by_header = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True)
    sized = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                           separators=MD_SEPARATORS, is_separator_regex=True)
    out = []
    for d in docs:
        for sec in by_header.split_text(d.page_content):
            path = " > ".join([d.metadata["title"]] + [sec.metadata[h] for h in ("h1", "h2", "h3") if h in sec.metadata])
            for piece in sized.split_text(sec.page_content):
                out.append(Document(page_content=f"{path}\n\n{piece}", metadata={**d.metadata, "section": path}))
    return out


def body(d: Document) -> str:
    """Chunk text without its heading-path prefix."""
    return d.page_content.split("\n\n", 1)[-1]


def unique(docs: list[Document]) -> list[Document]:
    """Drop results whose body repeats an earlier one (boilerplate shared by templated pages); keeps rank order.
    ponytail: exact match only. Near-duplicates (cosine > 0.95) also include distinct facts, so they're left to the M3 reranker."""
    seen = set()
    return [d for d in docs if not (body(d) in seen or seen.add(body(d)))]


def junk(d: Document) -> bool:
    """Bare fence leftovers from splitting, or printed numbers (embedding vectors in notebook output)."""
    b = body(d)
    return len(b.strip("`\n ")) < 20 or (len(b) > 100 and len(re.findall(r"[\d.,\-\s\[\]e]", b)) / len(b) > 0.8)


def build(subset: list[str] | None = None) -> list[Document]:
    docs = [Document(page_content=clean(d.page_content), metadata=d.metadata) for d in pages(subset)]
    return [c for c in chunk([d for d in docs if len(d.page_content) >= 300]) if not junk(c)]   # drop stub/redirect pages
