"""Chat service: multi-turn RAG over the LangChain docs, streamed as SSE, with citations and 👍/👎 feedback.

POST /chat      {"messages": [{"role": "user"|"assistant", "content": "..."}]}  ->  SSE: sources, token..., done | error
POST /feedback  {"id": "<turn id>", "rating": 1|-1, "comment": "..."}
GET  /          chat UI
"""
import json, time, uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from rag import corpus, generate, retrieve, store

DATA = store.ROOT / "data"
chunks = corpus.build()
db = store.get_store("full-v3", chunks)   # syncs on startup: only changed chunks re-embed
turns: dict[str, dict] = {}               # ponytail: in-memory, single user; turns.jsonl is the durable copy
app = FastAPI()


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
        try:
            question, history = req.messages[-1].content, [(m.role, m.content) for m in req.messages[:-1]]
            query = generate.condense(history, question) if history else question
            timings["condense"] = lap()
            pool = retrieve.search(db, query)
            timings["retrieve"] = lap()
            docs = retrieve.rerank(query, pool)[:4]
            timings["rerank"] = lap()
            sources = [{"source": d.metadata["source"], "section": d.metadata["section"], "url": url(d.metadata["source"])} for d in docs]
            yield sse("sources", sources)

            answer, model = "", "?"
            for chunk in generate.answer_chain.stream({"context": generate.format_context(docs), "question": query}):
                if chunk.content:
                    timings.setdefault("first_token", lap())
                    answer += chunk.content
                    yield sse("token", chunk.content)
                model = chunk.response_metadata.get("model_name", model)
            timings["total"] = lap()
            turn.update(query=query, sources=sources, answer=answer, model=model, timings=timings)
            yield sse("done", {"id": turn["id"], "query": query, "model": model, "timings": timings})
        except Exception as e:   # mid-stream: HTTP status is already sent, so report in-band
            turn.update(error=repr(e), timings=timings)
            yield sse("error", {"message": str(e)})
        turns[turn["id"]] = turn
        append(DATA / "turns.jsonl", turn)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/feedback")
def feedback(fb: Feedback):
    if fb.id not in turns:
        raise HTTPException(404, "unknown turn id")
    append(DATA / "feedback.jsonl", {**turns[fb.id], "rating": fb.rating, "comment": fb.comment, "at": time.time()})
    return {"ok": True}


@app.get("/")
def ui():
    return FileResponse(store.ROOT / "rag/chat.html")
