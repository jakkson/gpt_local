"""
ChromaDB vector store - stores document embeddings locally.
Handles chunking, embedding via Ollama, and similarity search.
"""

import hashlib
import logging
from pathlib import Path

import chromadb
from llama_index.core import Document as LIDocument
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)
from document_loader import Document

logger = logging.getLogger(__name__)

COLLECTION_NAME = "local_gpt_docs"


class NomicEmbedding(OllamaEmbedding):
    """Wraps OllamaEmbedding to add the prefixes that nomic-embed-text requires
    for correct asymmetric search (search_document: / search_query:)."""

    def _get_text_embedding(self, text: str) -> list[float]:
        return super()._get_text_embedding(f"search_document: {text}")

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._get_text_embedding(t) for t in texts]

    def _get_query_embedding(self, query: str) -> list[float]:
        return super()._get_text_embedding(f"search_query: {query}")

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await super()._aget_text_embedding(f"search_document: {text}")

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [await self._aget_text_embedding(t) for t in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await super()._aget_text_embedding(f"search_query: {query}")


def get_embed_model() -> NomicEmbedding:
    return NomicEmbedding(
        model_name=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class LocalVectorStore:
    """Manages the ChromaDB collection and LlamaIndex integration."""

    def __init__(self):
        self.chroma_client = get_chroma_client()
        self.collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
        self.embed_model = get_embed_model()

        Settings.embed_model = self.embed_model
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def _to_llama_doc(self, doc: Document) -> LIDocument:
        return LIDocument(text=doc.text, metadata=doc.metadata)

    def _get_existing_hashes(self) -> set[str]:
        """Fetch existing doc hashes from ChromaDB in safe batches."""
        total = self.collection.count()
        if total == 0:
            return set()
        hashes = set()
        batch_size = 5000
        for offset in range(0, total, batch_size):
            result = self.collection.get(
                limit=batch_size, offset=offset, include=[]
            )
            hashes.update(result["ids"])
        return hashes

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store one at a time. Returns count of new docs added."""
        existing = self._get_existing_hashes()
        added = 0

        for doc in documents:
            h = doc_hash(doc.text)
            if h in existing:
                continue

            try:
                li_doc = self._to_llama_doc(doc)
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                VectorStoreIndex.from_documents(
                    [li_doc],
                    storage_context=storage_context,
                    transformations=[self.splitter],
                    show_progress=False,
                )
                added += 1
                existing.add(h)
            except Exception as e:
                logger.warning(f"Failed to index {doc.metadata.get('filename')}: {e}")

        if added:
            logger.info(f"Added {added} documents ({self.collection.count()} total chunks)")
        return added

    def get_index(self) -> VectorStoreIndex:
        """Get a queryable index from the existing vector store."""
        return VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model,
        )

    def get_stats(self) -> dict:
        """Return stats about the current vector store."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "db_path": str(CHROMA_DB_DIR),
            "collection": COLLECTION_NAME,
        }

    def delete_chunks_by_filename_substrings(self, substrings: list[str]) -> int:
        """Remove chunks whose metadata filename contains any substring (case-insensitive)."""
        if not substrings:
            return 0
        needles = [s.lower() for s in substrings if s.strip()]
        if not needles:
            return 0
        ids_to_delete: list[str] = []
        batch_size = 2000
        total = self.collection.count()
        for offset in range(0, total, batch_size):
            result = self.collection.get(
                limit=min(batch_size, total - offset),
                offset=offset,
                include=["metadatas"],
            )
            for i, cid in enumerate(result["ids"]):
                meta = result["metadatas"][i] or {}
                fn = (meta.get("filename") or "").lower()
                if any(n in fn for n in needles):
                    ids_to_delete.append(cid)
        deleted = 0
        delete_batch = 400
        for i in range(0, len(ids_to_delete), delete_batch):
            chunk = ids_to_delete[i : i + delete_batch]
            self.collection.delete(ids=chunk)
            deleted += len(chunk)
        try:
            from hybrid_retrieval import invalidate_bm25_cache

            invalidate_bm25_cache()
        except ImportError:
            pass
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        logger.info(
            "Deleted %s chunks matching filename substrings: %s",
            deleted,
            needles,
        )
        return deleted

    def clear(self):
        """Delete all documents from the store."""
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        try:
            from hybrid_retrieval import invalidate_bm25_cache

            invalidate_bm25_cache()
        except ImportError:
            pass
        logger.info("Vector store cleared.")
