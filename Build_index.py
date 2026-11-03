"""Build (or load) the Chroma vectorstore -- the "Build Index" cell from the notebook.

Run once:  python build_index.py
"""
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
import os

PERSIST_DIR = "./chroma_db"
COLLECTION = "adaptive-rag"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The index is "about" these topics. The router uses this same description
# to decide whether a question belongs to the vectorstore.
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_index() -> Chroma:
    docs = [d for url in URLS for d in WebBaseLoader(url).load()]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=20)

    chunks = splitter.split_documents(docs)

    print(f"Loaded {len(docs)} pages -> {len(chunks)} chunks")

    return Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )


def get_retriever(k: int = 4):
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    build_index()
    print("Index built.")