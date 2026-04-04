"""
RAG engine - connects Ollama LLM with the local vector store
to answer questions grounded in your documents.
"""

import logging
from typing import Callable, Optional

from llama_index.core import Settings
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.llms.ollama import Ollama

from config import (
    HYBRID_FUSION_TOP_K,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
)
from hybrid_retrieval import build_retriever_query_engine
from rag_trace import build_trace_callback_manager
from vector_store import LocalVectorStore

logger = logging.getLogger(__name__)


def _apply_trace_callbacks(
    chat_engine,
    llm,
    trace_cm,
):
    """Point query engine, synthesizer, retriever(s), and LLM at the trace callback manager."""
    qe = chat_engine._query_engine
    ret = qe._retriever
    v = getattr(ret, "_vector", None)
    b = getattr(ret, "_bm25", None)
    old = {
        "qe": qe.callback_manager,
        "syn": qe._response_synthesizer.callback_manager,
        "chat": chat_engine.callback_manager,
        "llm": llm.callback_manager,
        "ret": ret.callback_manager,
        "vec": v.callback_manager if v else None,
        "bm25": b.callback_manager if b else None,
    }
    qe.callback_manager = trace_cm
    qe._response_synthesizer.callback_manager = trace_cm
    chat_engine.callback_manager = trace_cm
    llm.callback_manager = trace_cm
    ret.callback_manager = trace_cm
    if v is not None:
        v.callback_manager = trace_cm
    if b is not None:
        b.callback_manager = trace_cm
    return old


def _restore_trace_callbacks(chat_engine, llm, old: dict) -> None:
    qe = chat_engine._query_engine
    ret = qe._retriever
    v = getattr(ret, "_vector", None)
    b = getattr(ret, "_bm25", None)
    qe.callback_manager = old["qe"]
    qe._response_synthesizer.callback_manager = old["syn"]
    chat_engine.callback_manager = old["chat"]
    llm.callback_manager = old["llm"]
    ret.callback_manager = old["ret"]
    if v is not None and old.get("vec") is not None:
        v.callback_manager = old["vec"]
    if b is not None and old.get("bm25") is not None:
        b.callback_manager = old["bm25"]

SYSTEM_PROMPT = """You are a helpful AI assistant with access to the user's personal documents and emails.
Answer from the retrieved context. If it does not contain enough information, say so clearly instead of guessing.
Name the source file or email when you use a fact. If sources disagree, say that briefly."""


def _text_qa_prompt() -> PromptTemplate:
    head = (
        SYSTEM_PROMPT.strip()
        + "\n\n"
        "Prefer concise answers; use bullet points for lists when it helps readability.\n\n"
    )
    return PromptTemplate(
        head + DEFAULT_TEXT_QA_PROMPT_TMPL,
        prompt_type=PromptType.QUESTION_ANSWER,
    )


def get_llm() -> Ollama:
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
    )


class RAGEngine:
    """Query engine that combines local LLM with document retrieval."""

    def __init__(self):
        self.store = LocalVectorStore()
        self.llm = get_llm()
        Settings.llm = self.llm
        self._chat_engine = None

    def query(
        self,
        question: str,
        top_k: int | None = None,
        trace_sink: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """One-shot query: retrieve context + generate answer."""
        k = top_k if top_k is not None else HYBRID_FUSION_TOP_K
        query_engine = build_retriever_query_engine(
            self.store, self.llm, fusion_top_k=k, text_qa_template=_text_qa_prompt()
        )
        if trace_sink:
            trace_cm = build_trace_callback_manager(trace_sink)
            old = {
                "qe": query_engine.callback_manager,
                "syn": query_engine._response_synthesizer.callback_manager,
                "llm": self.llm.callback_manager,
            }
            ret = query_engine._retriever
            v = getattr(ret, "_vector", None)
            b = getattr(ret, "_bm25", None)
            old["ret"] = ret.callback_manager
            old["vec"] = v.callback_manager if v else None
            old["bm25"] = b.callback_manager if b else None
            query_engine.callback_manager = trace_cm
            query_engine._response_synthesizer.callback_manager = trace_cm
            self.llm.callback_manager = trace_cm
            ret.callback_manager = trace_cm
            if v is not None:
                v.callback_manager = trace_cm
            if b is not None:
                b.callback_manager = trace_cm
            trace_sink("Starting search…")
            try:
                response = query_engine.query(question)
            finally:
                query_engine.callback_manager = old["qe"]
                query_engine._response_synthesizer.callback_manager = old["syn"]
                self.llm.callback_manager = old["llm"]
                ret.callback_manager = old["ret"]
                if v is not None and old.get("vec") is not None:
                    v.callback_manager = old["vec"]
                if b is not None and old.get("bm25") is not None:
                    b.callback_manager = old["bm25"]
        else:
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

    def chat(
        self,
        message: str,
        top_k: int | None = None,
        trace_sink: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Conversational query with history condensation."""
        if self._chat_engine is None:
            k = top_k if top_k is not None else HYBRID_FUSION_TOP_K
            query_engine = build_retriever_query_engine(
                self.store,
                self.llm,
                fusion_top_k=k,
                text_qa_template=_text_qa_prompt(),
            )
            self._chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine,
                llm=self.llm,
            )

        if trace_sink:
            trace_cm = build_trace_callback_manager(trace_sink)
            old = _apply_trace_callbacks(self._chat_engine, self.llm, trace_cm)
            trace_sink("Rewriting question with chat context (if any)…")
            try:
                response = self._chat_engine.chat(message)
            finally:
                _restore_trace_callbacks(self._chat_engine, self.llm, old)
        else:
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
