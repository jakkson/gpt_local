"""
Streamlit web UI for Local GPT.
Chat with your documents + manage ingestion.
"""

import logging
import time
from pathlib import Path

import streamlit as st

from config import DOCUMENTS_DIR, OLLAMA_MODEL, SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Local GPT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { max-width: 1480px; margin: 0 auto; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 0.75rem;
        background: #16161e;
    }
    .source-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        font-size: 0.85em;
    }
    .stat-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .stat-number { font-size: 2em; font-weight: bold; color: #7c3aed; }
    .stat-label { font-size: 0.85em; color: #888; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_engine():
    from rag_engine import RAGEngine
    return RAGEngine()


@st.cache_resource
def get_vector_store():
    from vector_store import LocalVectorStore
    return LocalVectorStore()


# --- Sidebar ---
with st.sidebar:
    st.title("Local GPT")
    st.caption(f"Model: `{OLLAMA_MODEL}` via Ollama")

    try:
        store = get_vector_store()
        stats = store.get_stats()
        st.metric("Indexed Chunks", stats["total_chunks"])
    except Exception as e:
        st.error(f"Vector store error: {e}")
        stats = {}

    st.divider()

    tab_docs, tab_outlook, tab_settings = st.tabs(["Documents", "Outlook", "Settings"])

    # --- Document Ingestion Tab ---
    with tab_docs:
        st.subheader("Batch Ingest")

        uploaded_files = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            key="file_uploader",
        )

        folder_path = st.text_input(
            "Or enter folder path:",
            placeholder="/Users/you/Documents/reports",
        )

        col1, col2 = st.columns(2)
        with col1:
            ingest_btn = st.button("Ingest Files", type="primary", use_container_width=True)
        with col2:
            ingest_folder_btn = st.button("Ingest Folder", use_container_width=True)

        if ingest_btn and uploaded_files:
            with st.spinner("Processing uploaded files..."):
                from document_loader import Document, load_single_file

                DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
                docs = []
                for uf in uploaded_files:
                    save_path = DOCUMENTS_DIR / uf.name
                    save_path.write_bytes(uf.getbuffer())

                    doc = load_single_file(save_path)
                    if doc:
                        docs.append(doc)
                    else:
                        st.warning(f"Could not parse: {uf.name}")

                if docs:
                    store = get_vector_store()
                    added = store.add_documents(docs)
                    st.success(f"Ingested {added} documents!")
                    st.cache_resource.clear()
                    st.rerun()

        if ingest_folder_btn and folder_path:
            folder = Path(folder_path)
            if not folder.is_dir():
                st.error("Invalid folder path.")
            else:
                with st.spinner(f"Scanning {folder}..."):
                    from document_loader import load_documents, scan_directory

                    files = scan_directory(folder)
                    st.info(f"Found {len(files)} files.")

                    docs = list(load_documents([folder]))

                    if docs:
                        store = get_vector_store()
                        added = store.add_documents(docs)
                        st.success(f"Ingested {added} documents from folder!")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.warning("No parseable documents found.")

    # --- Outlook Tab ---
    with tab_outlook:
        st.subheader("Outlook 365 Emails")

        days_back = st.slider("Days back", 1, 365, 30)
        max_emails = st.slider("Max emails", 10, 500, 100)

        if st.button("Fetch & Ingest Emails", use_container_width=True):
            try:
                with st.spinner("Connecting to Outlook..."):
                    from outlook_client import OutlookClient

                    client = OutlookClient()
                    emails = client.fetch_emails(
                        days_back=days_back, max_emails=max_emails
                    )

                    if emails:
                        store = get_vector_store()
                        added = store.add_documents(emails)
                        st.success(f"Ingested {added} emails!")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.warning("No emails fetched.")
            except ValueError as e:
                st.error(str(e))
                st.info("Configure OUTLOOK_CLIENT_ID and OUTLOOK_TENANT_ID in your .env file.")
            except Exception as e:
                st.error(f"Outlook error: {e}")

    # --- Settings Tab ---
    with tab_settings:
        st.subheader("Database")
        if stats:
            st.json(stats)

        if st.button("Clear All Data", type="secondary"):
            store = get_vector_store()
            store.clear()
            st.cache_resource.clear()
            st.success("Vector store cleared!")
            st.rerun()

        if st.button("Reset Chat History"):
            st.session_state.messages = []
            try:
                engine = get_rag_engine()
                engine.reset_chat()
            except Exception:
                pass
            st.rerun()


# --- Main Chat Area (left) + live activity trace (right) ---
left, right = st.columns([2.1, 1], gap="medium")

with right:
    st.subheader("Search activity")
    st.caption("Steps for each question: retrieval, then the local LLM.")
    trace_panel = st.container(border=True)
    with trace_panel:
        trace_placeholder = st.empty()
        trace_placeholder.caption("Send a message to see live steps here.")

with left:
    col_title, col_new = st.columns([4, 1])
    with col_title:
        st.header("Chat with Your Documents")
    with col_new:
        st.write("")
        if st.button("New Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            try:
                engine = get_rag_engine()
                engine.reset_chat()
            except Exception:
                pass
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        score = f" | relevance: {src['score']}" if src.get("score") else ""
                        st.markdown(
                            f"**{src['filename']}**{score}\n\n"
                            f"_{src['text_preview']}_"
                        )

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        trace_lines: list[str] = []

        def trace_sink(line: str) -> None:
            trace_lines.append(line)
            trace_placeholder.markdown("\n\n".join(trace_lines))

        with st.chat_message("assistant"):
            with st.spinner("Working…"):
                try:
                    engine = get_rag_engine()
                    trace_sink("_Starting…_")
                    result = engine.chat(prompt, trace_sink=trace_sink)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                except Exception as e:
                    answer = (
                        f"Error: {e}\n\nMake sure Ollama is running (`ollama serve`) "
                        "and you have documents ingested."
                    )
                    sources = []
                    trace_sink(f"**Error:** `{e}`")

            st.markdown(answer)

            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for src in sources:
                        score = f" | relevance: {src['score']}" if src.get("score") else ""
                        st.markdown(
                            f"**{src['filename']}**{score}\n\n"
                            f"_{src['text_preview']}_"
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
