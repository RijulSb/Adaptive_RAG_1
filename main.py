"""Run the adaptive RAG graph on one question per route.

Usage:
    python main.py                       # runs the 3 demo questions
    python main.py "your question here"  # runs your own
"""
import sys

from Graph import app

DEMO_QUESTIONS = [
    # -> vectorstore
    "What are the types of agent memory?",
    # -> web_search
    "Who won the most recent Formula 1 race?",
    # -> llm_fallback
    "Hi there! How are you doing?",
]


def run(question: str) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("=" * 70)

    final = {}
    for step in app.stream({"question": question}):
        for node, update in step.items():
            print(f"[node] {node}")
            final.update(update)

    print(f"\nA: {final.get('generation', '(no answer)')}")


if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS
    for q in questions:
        run(q)