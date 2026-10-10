"""Grounded answer chain and LLM judge."""
import os
import re

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

rewrite_chain = ChatPromptTemplate.from_messages([("human", """Rewrite this question as a search query for the LangChain documentation.
Use the terms the docs would use (concept, class and function names) where you can infer them, and keep every specific name from the question.
Reply with the query only.

Question: {question}""")]) | llm | StrOutputParser()


condense_chain = ChatPromptTemplate.from_messages([("human", """Rewrite the user's last message as a standalone question, using the conversation
to resolve references ("it", "that one", "the second option"). Do not add facts, names or terms that are not in the conversation.
If the last message is already standalone, return it unchanged. Reply with the question only.

Conversation:
{history}

Last message: {question}""")]) | llm | StrOutputParser()


def _strip(text: str) -> str:
    text = text.replace("<|end|>", "")                          # gpt-oss special token leaks through the proxy
    return re.sub(r"[\x00-\x1f\u200b]+", " ", text).strip()   # so do NUL and zero-width spaces


def rewrite(question: str) -> str:
    return _strip(rewrite_chain.invoke({"question": question}))


def condense(history: list[tuple[str, str]], question: str) -> str:
    """history = [(role, text)]. Last 6 turns, long answers cut: references point to recent, short context."""
    h = "\n".join(f"{role}: {text[:600]}" for role, text in history[-6:])
    return _strip(condense_chain.invoke({"history": h, "question": question}))


def format_context(docs) -> str:
    return "\n\n---\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


def answer(question: str, docs) -> tuple[str, str]:
    """(text, model that actually answered). A gateway fallback can silently switch models."""
    msg = answer_chain.invoke({"context": format_context(docs), "question": question})
    return msg.content, msg.response_metadata.get("model_name", "?")


def judge(question: str, expected: str, answer: str) -> bool:
    return judge_chain.invoke({"question": question, "expected": expected, "answer": answer}).strip().upper().startswith("CORRECT")
