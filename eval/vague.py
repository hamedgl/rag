"""Vague queries: no single right answer, so the judge grades behaviour, not facts.
GOOD = says what the docs cover on that topic and asks which one is meant, or answers the most likely reading AND says
it assumed that reading. BAD = "Not in docs." for a documented topic, or one narrow answer as if it were the only one.
Run: uv run python -m eval.vague
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from eval.run import pmap
from rag import corpus, generate, retrieve, store

QUERIES = [
    "memory?",
    "how to stream",
    "my agent doesn't work",
    "best vector store",
    "middleware",
    "how do I make it faster",
    "tools not called",
    "deep agents vs langgraph",
]

JUDGE = ChatPromptTemplate.from_messages([("human", """A user asked a documentation assistant a VAGUE question: {q}
Assistant's answer:
<<<
{a}
>>>
GOOD if the answer either (a) briefly lays out the different things the docs cover for this topic and asks or invites
the user to say which one they mean, or (b) answers the most likely interpretation AND explicitly says that it assumed it.
BAD if it replies "Not in docs." (or similar refusal) although the topic is clearly documented, or if it answers one
narrow interpretation as if it were the only one, without noting the ambiguity.
Reply with one word: GOOD or BAD.""")]) | generate.judge_llm | StrOutputParser()


def main():
    db = store.get_store("full-v3", corpus.build())
    answers = pmap(lambda q: generate.answer(q, retrieve.rerank(q, retrieve.search(db, q))[:4])[0], QUERIES)
    verdicts = pmap(lambda x: JUDGE.invoke({"q": x[0], "a": x[1]}).strip().upper().startswith("GOOD"), list(zip(QUERIES, answers)))
    for q, a, ok in zip(QUERIES, answers, verdicts):
        print(f"{'✓' if ok else '✗'} {q:28} {a[:150].replace(chr(10), ' ')!r}")
    print(f"good: {sum(verdicts)}/{len(QUERIES)}")


if __name__ == "__main__":
    main()
