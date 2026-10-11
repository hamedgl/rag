"""Load, clean and chunk the LangChain docs (logic ported from the Phase 1 notebook, cleaner v2 + chunker v3)."""
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SRC = Path(__file__).resolve().parent.parent / "data/langchain-docs/src"
DOCS = SRC / "oss"          # LangChain, LangGraph, Deep Agents: sources are paths relative to here ("langchain/agents")
LANGSMITH = SRC / "langsmith"   # LangSmith (tracing, evals, deployment, Studio, Fleet): sources "langsmith/<page>"
OPENAPI = {"agent-server-api": LANGSMITH / "agent-server-openapi.json",        # published at docs.langchain.com/langsmith/<key>/...
           "smith-api": LANGSMITH / "langsmith-platform-openapi.json"}
V03 = SRC.parent.parent / "langchain-v03/docs/docs"   # v0.3 docs (langchain repo, branch v0.3; concepts, how_to, tutorials)
V03_DIRS = ("concepts", "how_to", "tutorials")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
SNIPPET_IMPORT = re.compile(r"""^import (\w+) from ['"](/snippets/[^'"]+\.mdx)['"];?[ \t]*$""", re.M)
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


def inline_snippets(text: str, depth: int = 0) -> str:
    """Replace <Name /> with the body of the snippet `Name` was imported from. Pages share content through snippets
    (788 imports in oss alone); without this, the cleaner drops the import and that content never gets indexed."""
    if depth > 3:   # snippets import snippets; cap against cycles
        return text
    for name, path in SNIPPET_IMPORT.findall(text):
        f = SRC / path.lstrip("/")
        if f.exists():
            body = inline_snippets(FRONTMATTER.sub("", f.read_text(), count=1), depth + 1)
            text = re.sub(rf"<{name}(?:\s[^>]*)?/>", lambda m: body, text)
    return text


def load(path: Path) -> Document:
    raw = path.read_text()
    fm = FRONTMATTER.match(raw)
    title = re.search(r"^title:\s*(.+)$", fm.group(1), re.M) if fm else None
    rel = str(path.relative_to(DOCS) if path.is_relative_to(DOCS) else Path("langsmith") / path.relative_to(LANGSMITH)).removesuffix(".mdx")
    raw = inline_snippets(raw[fm.end():] if fm else raw)
    return Document(page_content=raw,
                    metadata={"source": rel, "title": title.group(1).strip('"') if title else rel, "version": "v1"})


def notebook_text(path: Path) -> str:
    """Markdown cells as-is, code cells as ```python fences. Outputs dropped (979 outputs / 2.3 MB of printed
    results, logs and base64 images in v0.3: noise next to the code that produced them)."""
    cells = json.loads(path.read_text())["cells"]
    return "\n\n".join("".join(c["source"]) if c["cell_type"] == "markdown" else f"```python\n{''.join(c['source'])}\n```"
                        for c in cells if c["cell_type"] in ("markdown", "code") and "".join(c["source"]).strip())


def load_v03(path: Path) -> Document:
    text = notebook_text(path) if path.suffix == ".ipynb" else path.read_text()
    fm = FRONTMATTER.match(text)
    text = text[fm.end():] if fm else text
    rel = "v0.3/" + str(path.relative_to(V03).with_suffix(""))
    title = re.search(r"^# (.+)$", text, re.M)
    return Document(page_content=text, metadata={"source": rel, "title": title.group(1).strip() if title else rel, "version": "v0.3"})


def pages(subset: list[str] | None = None) -> list[Document]:
    """All oss pages except the JS duplicates, plus LangSmith and the v0.3 docs; or only `subset` (oss)."""
    if subset:
        paths = [DOCS / f"{s}.mdx" for s in subset]
    else:
        paths = sorted(p for p in DOCS.rglob("*.mdx") if "javascript" not in p.relative_to(DOCS).parts) + sorted(LANGSMITH.rglob("*.mdx"))
    v03 = sorted(p for d in V03_DIRS for p in (V03 / d).rglob("*") if p.suffix in (".ipynb", ".mdx", ".md")) if not subset else []
    return [load(p) for p in paths] + [load_v03(p) for p in v03]


def _paramfield(m: re.Match) -> str:
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0)))
    name = attrs.get("path") or attrs.get("body") or attrs.get("query") or "?"
    extra = ", ".join(f"{k}: {attrs[k]}" for k in ("type", "default") if k in attrs)
    return f"Parameter `{name}`" + (f" ({extra})" if extra else "") + ":"


HTML_TAG = r"</?(?:a|div|span|br|p|ul|ol|li|table|thead|tbody|tr|td|th|code|img)\b[^>]*>"


def _prose(text: str) -> str:
    """MDX/HTML cleanup for text outside code fences (in code, `import x` and `<Foo>` are real code)."""
    text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.S)                                 # JSX comments: not rendered
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)                                    # HTML comments: not rendered either
    text = re.sub(r"<ParamField\s[^>]*>", _paramfield, text)                              # keep name + default before tags go
    text = re.sub(r'<[A-Z][A-Za-z]*\s[^>]*?\btitle="([^"]*)"[^>]*>', r"\n**\1**\n", text) # Tab/Accordion/Card/Step titles
    text = re.sub(r'<img\s[^>]*?\balt="([^"]*)"[^>]*>', r"\1", text)                      # images -> alt text
    text = re.sub(r"^import .*$\n?", "", text, flags=re.M)                                # snippet imports
    text = re.sub(r"</?[A-Z][A-Za-z]*(?:\s[^>]*)?/?>", "", text)                          # JSX tags, keep inner text
    text = re.sub(HTML_TAG, "", text)                                                     # HTML tags, keep inner text
    return re.sub(r"@\[(`?[^\]]+`?)\]", r"\1", text)                                     # @[`X`] -> `X`


def clean(text: str) -> str:
    text = re.sub(r"^[ \t]*:::js\n.*?^[ \t]*:::[ \t]*\n?", "", text, flags=re.S | re.M)   # JS duplicate content
    text = re.sub(r"^[ \t]*:::(?:info|note|tip|important|danger|caution|warning)\b[^\n]*\n?", "", text, flags=re.M)  # v0.3 admonition markers
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


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _type(schema: dict) -> str:
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in schema or "oneOf" in schema:
        return " | ".join(_type(x) for x in schema.get("anyOf") or schema.get("oneOf"))
    if schema.get("type") == "array":
        return f"array[{_type(schema.get('items', {}))}]"
    return schema.get("type", "object")


def _fields(schema: dict, spec: dict, depth: int = 0) -> list[str]:
    """Top-level properties of a request body schema: '- name (type, required): description'. Resolves $ref and merges
    allOf (schemas that extend a base schema, e.g. RunCreateStreamingStateful = RunCreateStateful + on_disconnect)."""
    if "$ref" in schema:
        schema = spec["components"]["schemas"].get(schema["$ref"].rsplit("/", 1)[-1], {})
    if "allOf" in schema and depth < 3:
        return [f for part in schema["allOf"] for f in _fields(part, spec, depth + 1)]
    req = set(schema.get("required", []))
    return [f"- `{n}` ({_type(p)}{', required' if n in req else ''}){': ' + p['description'].strip() if p.get('description') else ''}"
            for n, p in schema.get("properties", {}).items()]


def api_chunks() -> list[Document]:
    """One chunk per OpenAPI operation (632 across the two LangSmith specs). Built directly, not header-split:
    an endpoint must stay together with its parameters. ponytail: request-body $refs resolved one level, capped at 3k chars."""
    out = []
    for key, path in OPENAPI.items():
        spec = json.loads(path.read_text())
        api = spec["info"]["title"] + " API"
        for route, methods in spec["paths"].items():
            for method, op in methods.items():
                if method not in ("get", "post", "put", "patch", "delete"):
                    continue
                tag, summary = (op.get("tags") or ["other"])[0], op.get("summary") or op.get("operationId", route)
                parts = [f"`{method.upper()} {route}`", op.get("description", "").strip()]
                params = [f"- `{x['name']}` ({x.get('in')}{', required' if x.get('required') else ''}, {_type(x.get('schema', {}))})"
                          + (f": {x['description'].strip()}" if x.get("description") else "") for x in op.get("parameters", []) if "name" in x]
                if params:
                    parts.append("Parameters:\n" + "\n".join(params))
                body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
                if body:
                    fields = _fields(body, spec)
                    parts.append("Request body:\n" + "\n".join(fields) if fields else f"Request body: {_type(body)}")
                parts.append("Responses: " + ", ".join(f"{c} {r.get('description', '')}".strip() for c, r in op.get("responses", {}).items()))
                section = f"{api} > {tag} > {summary}"
                text = (section + "\n\n" + "\n\n".join(x for x in parts if x))[:3000]
                out.append(Document(page_content=text, metadata={"source": f"langsmith/{key}/{_slug(tag)}/{_slug(summary)}", "title": api,
                                                                 "section": section, "version": "v1", "kind": "api"}))
    return out


def build(subset: list[str] | None = None) -> list[Document]:
    docs = [Document(page_content=clean(d.page_content), metadata=d.metadata) for d in pages(subset)]
    chunks = [c for c in chunk([d for d in docs if len(d.page_content) >= 300]) if not junk(c)]   # drop stub/redirect pages
    return chunks + (api_chunks() if not subset else [])
