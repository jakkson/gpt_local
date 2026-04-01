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


def get_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(
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

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store one at a time. Returns count of new docs added."""
        existing_ids = set(self.collection.get()["ids"]) if self.collection.count() > 0 else set()
        added = 0

        for doc in documents:
            h = doc_hash(doc.text)
            if h in existing_ids:
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

    def clear(self):
        """Delete all documents from the store."""
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        logger.info("Vector store cleared.")
