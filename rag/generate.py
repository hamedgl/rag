"""Grounded answer chain and LLM judge."""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag import store  # noqa: F401  (loads .env)

def _chat(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], temperature=0)

llm = _chat(os.environ["LLM_MODEL"])
judge_llm = _chat(os.environ.get("JUDGE_MODEL", os.environ["LLM_MODEL"]))   # keep the grader fixed when swapping the generator

SYSTEM = """You answer questions about LangChain using ONLY the context below.
Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "Not in docs."
- If sources in the context disagree, say so and cite each.
- Cite the source of every claim in square brackets, e.g. [langchain/overview].

Context:
{context}"""

answer_chain = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{question}")]) | llm   # no parser: keep model_name

judge_chain = ChatPromptTemplate.from_messages([("human", """Grade an answer against the expected answer.
Question: {question}
Expected: {expected}
Answer: {answer}

CORRECT if the answer contains the expected facts (extra correct detail is fine; if expected says "Not in docs", the answer must decline).
WRONG if it misses a key fact, contradicts the expected answer, or answers when it should decline.
Reply with one word: CORRECT or WRONG.""")]) | judge_llm | StrOutputParser()


def format_context(docs) -> str:
    return "\n\n---\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


def answer(question: str, docs) -> tuple[str, str]:
    """(text, model that actually answered). A gateway fallback can silently switch models."""
    msg = answer_chain.invoke({"context": format_context(docs), "question": question})
    return msg.content, msg.response_metadata.get("model_name", "?")


def judge(question: str, expected: str, answer: str) -> bool:
    return judge_chain.invoke({"question": question, "expected": expected, "answer": answer}).strip().upper().startswith("CORRECT")
