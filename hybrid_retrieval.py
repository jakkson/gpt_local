"""
Hybrid retrieval: dense (Chroma + embeddings) + BM25, merged with reciprocal rank fusion.
Improves recall for names, product terms, and manual sections that pure embedding search misses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import chromadb

if TYPE_CHECKING:
    from llama_index.core.prompts import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from config import (
    HYBRID_BM25_CANDIDATES,
    HYBRID_BM25_REBUILD_MIN_DELTA,
    HYBRID_ENABLED,
    HYBRID_FUSION_TOP_K,
    HYBRID_RRF_K,
    HYBRID_VECTOR_CANDIDATES,
)
from vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

_bm25_retriever: Optional[BM25Retriever] = None
_bm25_source_count: int = -1


def invalidate_bm25_cache() -> None:
    """Call after a bulk clear; next query rebuilds BM25 from Chroma."""
    global _bm25_retriever, _bm25_source_count
    _bm25_retriever = None
    _bm25_source_count = -1


def load_text_nodes_from_chroma(
    collection: chromadb.Collection, batch_size: int = 2000
) -> list[TextNode]:
    """Load all chunk texts from Chroma as TextNodes for BM25."""
    nodes: list[TextNode] = []
    total = collection.count()
    if total == 0:
        return nodes
    for offset in range(0, total, batch_size):
        r = collection.get(
            limit=min(batch_size, total - offset),
            offset=offset,
            include=["documents", "metadatas"],
        )
        for i, node_id in enumerate(r["ids"]):
            doc = r["documents"][i]
            if not doc or not str(doc).strip():
                continue
            meta = r["metadatas"][i] or {}
            nodes.append(TextNode(text=str(doc), id_=node_id, metadata=dict(meta)))
    return nodes


def reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]],
    k: int,
    top_n: int,
) -> list[NodeWithScore]:
    """RRF: merge multiple ranked lists without score normalization."""
    scores: dict[str, float] = {}
    id_to_nws: dict[str, NodeWithScore] = {}
    for results in ranked_lists:
        for rank, nws in enumerate(results):
            nid = nws.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
            id_to_nws[nid] = nws
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    out: list[NodeWithScore] = []
    for nid in sorted_ids[:top_n]:
        base = id_to_nws[nid]
        out.append(NodeWithScore(node=base.node, score=scores[nid]))
    return out


class RRFHybridRetriever(BaseRetriever):
    """Runs vector + BM25 retrievers and fuses with reciprocal rank fusion."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        rrf_k: int = 60,
        fusion_top_k: int = 8,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(callback_manager=callback_manager)
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k
        self._fusion_top_k = fusion_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        v = self._vector.retrieve(query_bundle)
        b = self._bm25.retrieve(query_bundle)
        return reciprocal_rank_fusion([v, b], k=self._rrf_k, top_n=self._fusion_top_k)


def _get_or_build_bm25(collection: chromadb.Collection) -> Optional[BM25Retriever]:
    global _bm25_retriever, _bm25_source_count
    n = collection.count()
    if n == 0:
        return None
    if _bm25_retriever is not None and _bm25_source_count == n:
        return _bm25_retriever
    if (
        _bm25_retriever is not None
        and n > _bm25_source_count
        and (n - _bm25_source_count) < HYBRID_BM25_REBUILD_MIN_DELTA
    ):
        return _bm25_retriever
    logger.info("Building BM25 index from Chroma…")
    nodes = load_text_nodes_from_chroma(collection)
    if not nodes:
        return None
    _bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=HYBRID_BM25_CANDIDATES,
    )
    _bm25_source_count = n
    logger.info("BM25 index ready (%s chunks).", len(nodes))
    return _bm25_retriever


def build_retriever_query_engine(
    store: LocalVectorStore,
    llm,
    fusion_top_k: int,
    text_qa_template: Optional["PromptTemplate"] = None,
):
    """
    Query engine using hybrid RRF when enabled and BM25 can be built; else dense-only.
    """
    qa_kw = {}
    if text_qa_template is not None:
        qa_kw["text_qa_template"] = text_qa_template

    index = store.get_index()
    if not HYBRID_ENABLED:
        return index.as_query_engine(
            similarity_top_k=fusion_top_k, llm=llm, **qa_kw
        )

    bm25 = _get_or_build_bm25(store.collection)
    if bm25 is None:
        return index.as_query_engine(
            similarity_top_k=fusion_top_k, llm=llm, **qa_kw
        )

    vector_retriever = index.as_retriever(similarity_top_k=HYBRID_VECTOR_CANDIDATES)
    hybrid = RRFHybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25,
        rrf_k=HYBRID_RRF_K,
        fusion_top_k=fusion_top_k,
    )
    return RetrieverQueryEngine.from_args(retriever=hybrid, llm=llm, **qa_kw)
