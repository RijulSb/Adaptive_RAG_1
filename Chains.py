"""All the LLM-powered components used by the graph.

Mapping to the diagram:
  question_router        -> "Query Analysis"
  retrieval_grader       -> "Relevant to question?"
  rag_chain              -> "Generate"
  hallucination_grader   -> "Hallucinations?"
  answer_grader          -> "Answers question?"
  question_rewriter      -> improves the query before "Web search"
  llm_fallback_chain     -> "anything else" branch (answer with LLM only)
"""
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatCohere(model=os.getenv("COHERE_API_KEY", "command-r-plus"), temperature=0)


# --------------------------------------------------------------------------
# 1. Query analysis / router
# --------------------------------------------------------------------------
class RouteQuery(BaseModel):
    """Route a user question to the most relevant data source."""

    datasource: Literal["vectorstore", "web_search", "llm_fallback"] = Field(
        description=(
            "vectorstore for LLM agents, prompt engineering and adversarial "
            "attacks on LLMs; web_search for current events or recent info; "
            "llm_fallback for anything else (chit-chat, general knowledge)."
        )
    )


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at routing a user question.\n"
            "- The vectorstore contains documents about LLM agents, prompt "
            "engineering, and adversarial attacks on LLMs. Use it for those topics.\n"
            "- Use web_search for questions about current events or anything "
            "that needs up-to-date information.\n"
            "- Otherwise use llm_fallback.",
        ),
        ("human", "{question}"),
    ]
)
question_router = router_prompt | llm.with_structured_output(RouteQuery)


# --------------------------------------------------------------------------
# 2. Retrieval grader (is this document relevant?)
# --------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    """Binary relevance check on a retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        description="Is the document relevant to the question? 'yes' or 'no'"
    )


retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing the relevance of a retrieved document "
            "to a user question. If the document contains keywords or semantic "
            "meaning related to the question, grade it as relevant. This is a "
            "lenient check meant only to filter out clearly erroneous retrievals. "
            "Answer 'yes' or 'no'.",
        ),
        ("human", "Document:\n\n{document}\n\nQuestion: {question}"),
    ]
)
retrieval_grader = retrieval_prompt | llm.with_structured_output(GradeDocuments)

# --------------------------------------------------------------------------
# 3. RAG generation
# --------------------------------------------------------------------------
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use ONLY the "
            "retrieved context below to answer. If the context does not contain "
            "the answer, say you don't know. Use at most three sentences.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
rag_chain = rag_prompt | llm | StrOutputParser()


# --------------------------------------------------------------------------
# 4. Hallucination grader (is the answer grounded in the documents?)
# --------------------------------------------------------------------------
class GradeHallucinations(BaseModel):
    """Is the generation grounded in the provided facts?"""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts: 'yes' or 'no'"
    )


hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an LLM generation is grounded "
            "in / supported by a set of retrieved facts. Answer 'yes' if it is "
            "grounded, 'no' if it contains unsupported claims.",
        ),
        ("human", "Facts:\n\n{documents}\n\nLLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | llm.with_structured_output(
    GradeHallucinations
)


# --------------------------------------------------------------------------
# 5. Answer grader (does the answer actually address the question?)
# --------------------------------------------------------------------------
class GradeAnswer(BaseModel):
    """Does the answer resolve the question?"""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question: 'yes' or 'no'"
    )


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether an answer addresses / resolves "
            "a question. Answer 'yes' or 'no'.",
        ),
        ("human", "Question: {question}\n\nAnswer: {generation}"),
    ]
)
answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)

# --------------------------------------------------------------------------
# 6. Question rewriter (optimizes the query for web search)
# --------------------------------------------------------------------------
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite input questions into a better version optimized for "
            "web search. Reason about the underlying semantic intent. "
            "Return ONLY the rewritten question.",
        ),
        ("human", "Initial question: {question}\n\nImproved question:"),
    ]
)
question_rewriter = rewrite_prompt | llm | StrOutputParser()

# --------------------------------------------------------------------------
# 7. Fallback: answer directly with the LLM
# --------------------------------------------------------------------------
fallback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer concisely."),
        ("human", "{question}"),
    ]
)
llm_fallback_chain = fallback_prompt | llm | StrOutputParser()


