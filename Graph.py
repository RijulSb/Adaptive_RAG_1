
"""LangGraph implementation of the Adaptive RAG diagram.

            ┌──────────── question ────────────┐
            │         (query analysis)         │
   [vectorstore]     [web_search]       [llm_fallback]
        │                 │                   │
     retrieve             │              answer w/ LLM -> END
        │                 │
   grade documents        │
     │         │          │
  relevant   none relevant│
     │         └─> rewrite query -> web search
     │                         │
     └──────────> generate <───┘
                     │
            hallucination check ──(not grounded)──> generate (retry)
                     │ grounded
              answers question? ──(no)──> rewrite query -> web search
                     │ yes
                    END
"""
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from Build_index import get_retriever
from Chains import (
    answer_grader,
    hallucination_grader,
    llm_fallback_chain,
    question_rewriter,
    question_router,
    rag_chain,
    retrieval_grader,
)

load_dotenv()

MAX_GENERATIONS = 3  # hallucination retries
MAX_REWRITES = 2  # query rewrite -> web search loops

retriever = get_retriever()
web_search_tool = TavilySearch(max_results=3)


class GraphState(TypedDict, total=False):
    question: str
    generation: str
    documents: List[Document]
    generations: int  # how many times we've generated
    rewrites: int  # how many times we've rewritten the query


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

# retrieve docs
def retrieve(state: GraphState) -> GraphState:
    docs = retriever.invoke(state['question'])
    return {"documents": docs}


def grade_documents(state: GraphState) -> GraphState:
    """Keep only documents the grader marks as relevant."""
    kept = []

    for d in state['documents']:
        score = retrieval_grader.invoke({"question": state['question'], "document": d.page_content})
        if score.binary_score == "yes":
            kept.append[d]

    print(f"   kept {len(kept)}/{len(state['documents'])} documents")
    return {"documents": kept}


def transform_query(state: GraphState) -> GraphState:
    better = question_rewriter.invoke({"question": state["question"]})
    print(f"   rewritten query: {better}")
    return {"question": better, "rewrites": state.get("rewrites", 0) + 1}


def web_search(state: GraphState) -> GraphState:
    result = web_search_tool.invoke({"query": state['question']})
    hits = result['results'] if isinstance(result, dict) else result
    docs = [
        Document(page_content=h["content"], metadata={"source": h.get("url")})
        for h in hits
    ]
    return {"documents": docs}


def generate(state: GraphState) -> GraphState:
    context = "\n\n".join(d.page_content for d in state['documents'])
    answer = rag_chain.invoke({"question": state["question"], "context": context})
    return {"generation": answer, "generations": state.get("generations", 0) + 1}


def llm_fallback(state: GraphState) -> GraphState:
    return {"generation": llm_fallback_chain.invoke({"question": state["question"]})}


# --------------------------------------------------------------------------
# Conditional edges
# --------------------------------------------------------------------------
def route_question(state: GraphState) -> str:
    route = question_router.invoke({"question": state["question"]})
    print(f"---ROUTE: {route.datasource}---")
    return route.datasource


def decide_after_grading(state: GraphState) -> str:
    if not state["documents"]:
        print("---NO RELEVANT DOCS -> rewrite + web search---")
        return "transform_query"
    return "generate"


def grade_generation(state: GraphState) -> str:
    docs_text = "\n\n".join(d.page_content for d in state["documents"])

    grounded = hallucination_grader.invoke({"documents": docs_text, "generation": state['generation']})

    if grounded.binary_score == "no":
        if state.get("generations", 0) >= MAX_GENERATIONS:
            print("---STILL HALLUCINATING, MAX RETRIES -> giving up---")
            return "end"
        print("---NOT GROUNDED -> regenerate---")
        return "generate"

    useful = answer_grader.invoke({"question": state['question'], "generation": state['generation']})

    if useful.binary_score == "yes":
        print("---GROUNDED + ANSWERS QUESTION---")
        return "end"

    if state.get("rewrites", 0) >= MAX_REWRITES:
        print("---ANSWER NOT USEFUL, MAX REWRITES -> giving up---")
        return "end"
    print("---GROUNDED BUT DOESN'T ANSWER -> rewrite + web search---")
    return "transform_query"


#------------------------------------------
#---------- WIRE USING LANGGRAPH ----------
#------------------------------------------


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("transform_query", transform_query)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)
    g.add_node("llm_fallback", llm_fallback)

    #Query Analysis
    g.add_conditional_edges(
        START,
        route_question,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
            "llm_fallback": "llm_fallback"
        },
    )
    # Vectorstore branch
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"transform_query": "transform_query", "generate": "generate"},
    )

    # Web search branch
    g.add_edge("transform_query", "web_search")
    g.add_edge("web_search", "generate")

    # Self-reflection after generation
    g.add_conditional_edges(
        "generate",
        grade_generation,
        {"generate": "generate", "transform_query": "transform_query", "end": END},
    )

    g.add_edge("llm_fallback", END)
    return g.compile()

app = build_graph()



