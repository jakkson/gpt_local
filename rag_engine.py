"""
RAG engine - connects Ollama LLM with the local vector store
to answer questions grounded in your documents.
"""

import logging

from llama_index.core import Settings
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.llms.ollama import Ollama

from config import HYBRID_FUSION_TOP_K, OLLAMA_BASE_URL, OLLAMA_MODEL
from hybrid_retrieval import build_retriever_query_engine
from vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to the user's personal documents and emails.
Answer questions based on the retrieved context. If the context doesn't contain
relevant information, say so honestly rather than making things up.
When referencing information, mention which document or email it came from."""


def get_llm() -> Ollama:
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )


class RAGEngine:
    """Query engine that combines local LLM with document retrieval."""

    def __init__(self):
        self.store = LocalVectorStore()
        self.llm = get_llm()
        Settings.llm = self.llm
        self._chat_engine = None

    def query(self, question: str, top_k: int | None = None) -> dict:
        """One-shot query: retrieve context + generate answer."""
        k = top_k if top_k is not None else HYBRID_FUSION_TOP_K
        query_engine = build_retriever_query_engine(self.store, self.llm, fusion_top_k=k)
        response = query_engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "unknown"),
                "score": round(node.score, 3) if node.score else None,
                "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            })

        return {
            "answer": str(response),
            "sources": sources,
        }

    def chat(self, message: str, top_k: int | None = None) -> dict:
        """Conversational query with history condensation."""
        if self._chat_engine is None:
            k = top_k if top_k is not None else HYBRID_FUSION_TOP_K
            query_engine = build_retriever_query_engine(self.store, self.llm, fusion_top_k=k)
            self._chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine,
                llm=self.llm,
            )

        response = self._chat_engine.chat(message)

        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                sources.append({
                    "filename": node.metadata.get("filename", "unknown"),
                    "score": round(node.score, 3) if node.score else None,
                    "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                })

        return {
            "answer": str(response),
            "sources": sources,
        }

    def reset_chat(self):
        """Clear conversation history."""
        self._chat_engine = None

    def get_stats(self) -> dict:
        stats = self.store.get_stats()
        stats["llm_model"] = OLLAMA_MODEL
        return stats
