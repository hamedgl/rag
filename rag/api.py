"""Chat service: multi-turn RAG over the LangChain docs, streamed as SSE, with citations and 👍/👎 feedback.

POST /chat      {"messages": [{"role": "user"|"assistant", "content": "..."}]}  ->  SSE: sources, token..., done | error
POST /feedback  {"id": "<turn id>", "rating": 1|-1, "comment": "..."}
GET  /          chat UI
"""
import json, os, time, uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from rag import corpus, generate, retrieve, store

DATA = store.ROOT / "data"
chunks = corpus.build()
db = store.get_store("full-v3", chunks)   # syncs on startup: only changed chunks re-embed
turns: dict[str, dict] = {}               # ponytail: in-memory, single user; turns.jsonl is the durable copy
app = FastAPI()
lf = get_client()   # Langfuse; a no-op client when LANGFUSE_* keys are unset


def traced(span) -> dict:
    """LangChain config that nests LLM calls under `span`. Explicit parent, not contextvars: Starlette runs
    a streaming generator on different threads between yields, so a 'current span' would be lost."""
    return {"callbacks": [CallbackHandler(trace_context={"trace_id": span.trace_id, "parent_span_id": span.id})]}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Chat(BaseModel):
    messages: list[Message]


class Feedback(BaseModel):
    id: str
    rating: Literal[1, -1]
    comment: str = ""


def url(source: str) -> str:
    if source.startswith("langsmith/"):
        return "https://docs.langchain.com/" + source
    return "https://docs.langchain.com/oss/python/" + source.removeprefix("python/")


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def append(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.post("/chat")
def chat(req: Chat):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(400, "last message must be from the user")

    def events():
        t0 = time.perf_counter()
        lap = lambda: round(time.perf_counter() - t0, 2)
        timings, turn = {}, {"id": uuid.uuid4().hex[:12], "messages": [m.model_dump() for m in req.messages]}
        root = lf.start_observation(name="chat", as_type="chain", input={"messages": turn["messages"]})
        turn["trace_id"] = root.trace_id
        try:
            question, history = req.messages[-1].content, [(m.role, m.content) for m in req.messages[:-1]]
            query = generate.condense(history, question, config=traced(root)) if history else question
            timings["condense"] = lap()

            span = root.start_observation(name="retrieve", as_type="retriever", input=query)
            pool = retrieve.search(db, query)
            span.update(output=[d.metadata["section"] for d in pool])
            span.end()
            timings["retrieve"] = lap()

            reranker = os.environ.get("RERANK_MODEL") or "generator LLM"
            span = root.start_observation(name="rerank", input={"query": query, "candidates": len(pool)}, metadata={"reranker": reranker})
            docs = retrieve.rerank(query, pool)[:4]
            sources = [{"source": d.metadata["source"], "section": d.metadata["section"], "url": url(d.metadata["source"])} for d in docs]
            span.update(output=[s["section"] for s in sources])
            span.end()
            timings["rerank"] = lap()
            yield sse("sources", sources)

            answer, model = "", "?"
            for chunk in generate.answer_chain.stream({"context": generate.format_context(docs), "question": query}, config=traced(root)):
                if chunk.content:
                    timings.setdefault("first_token", lap())
                    answer += chunk.content
                    yield sse("token", chunk.content)
                model = chunk.response_metadata.get("model_name", model)
            timings["total"] = lap()
            turn.update(query=query, sources=sources, answer=answer, model=model, timings=timings)
            root.update(output=answer, metadata={"query": query, "model": model, "timings": timings, "turn_id": turn["id"]})
            yield sse("done", {"id": turn["id"], "query": query, "model": model, "timings": timings})
        except Exception as e:   # mid-stream: HTTP status is already sent, so report in-band
            turn.update(error=repr(e), timings=timings)
            root.update(level="ERROR", status_message=repr(e), metadata={"timings": timings})
            yield sse("error", {"message": str(e)})
        finally:
            root.end()
        turns[turn["id"]] = turn
        append(DATA / "turns.jsonl", turn)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/feedback")
def feedback(fb: Feedback):
    if fb.id not in turns:
        raise HTTPException(404, "unknown turn id")
    append(DATA / "feedback.jsonl", {**turns[fb.id], "rating": fb.rating, "comment": fb.comment, "at": time.time()})
    lf.create_score(name="user_feedback", value=fb.rating, data_type="NUMERIC", trace_id=turns[fb.id]["trace_id"], comment=fb.comment or None)
    return {"ok": True}


@app.get("/")
def ui():
    return FileResponse(store.ROOT / "rag/chat.html")
