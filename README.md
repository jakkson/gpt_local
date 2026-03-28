# Local GPT - Private AI That Learns Your Documents & Emails

A fully local, private AI assistant that runs on your Mac using **Ollama + Llama 3.2**. It indexes your documents and Outlook 365 emails, then lets you chat with them — all data stays on your machine.

## Features

- **100% Local** — LLM runs on your Mac via Ollama. No data leaves your machine.
- **Batch Document Ingestion** — Drop entire folders of PDFs, DOCX, XLSX, CSV, TXT, PPTX, HTML, emails, and more
- **Outlook 365 Integration** — Fetch and index your work emails via Microsoft Graph API
- **Smart Retrieval (RAG)** — Answers grounded in your actual documents with source citations
- **Web UI** — Clean Streamlit chat interface with drag-and-drop upload
- **CLI Tool** — `python ingest.py` for scripted/batch workflows
- **File Watcher** — Auto-ingest new files dropped into a folder

## Quick Start

### 1. Start Ollama & Pull Models

```bash
# Start Ollama (if not already running)
ollama serve

# Pull the LLM and embedding model (in another terminal)
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install Python Dependencies

```bash
cd /Users/poormanairm4/Dropbox/Documents/local_gpt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env if you want to change models or paths
```

### 4. Ingest Documents

```bash
# Ingest a folder of documents
python ingest.py /path/to/your/documents

# Ingest multiple paths
python ingest.py ~/Documents/reports ~/Desktop/notes.pdf

# Ingest the default ./documents folder
mkdir -p documents
# Copy your files into ./documents/
python ingest.py
```

### 5. Launch the Web UI

```bash
streamlit run app.py
```

Open http://localhost:8501 and start chatting with your documents!

## Batch Ingestion (CLI)

The `ingest.py` CLI supports many workflows:

```bash
# Ingest a single folder
python ingest.py /path/to/docs

# Ingest multiple folders and files at once
python ingest.py /path/to/folder1 /path/to/folder2 /path/to/file.pdf

# Re-ingest from scratch (clears database first)
python ingest.py --clear /path/to/docs

# Watch a folder for new files (auto-ingest)
python ingest.py --watch /path/to/docs

# Show database stats
python ingest.py --stats

# Fetch Outlook emails + local docs
python ingest.py /path/to/docs --outlook --days 60
```

### Supported File Types

| Type | Extensions |
|------|-----------|
| Documents | `.pdf`, `.docx`, `.doc`, `.txt`, `.rtf`, `.md` |
| Spreadsheets | `.xlsx`, `.xls`, `.csv` |
| Presentations | `.pptx` |
| Web | `.html`, `.htm` |
| Email | `.eml` |
| Data | `.json`, `.xml`, `.log` |

## Outlook 365 Setup

To connect your work email:

### 1. Register an Azure App

1. Go to [Azure Portal](https://portal.azure.com) → Azure Active Directory → App registrations
2. Click **New registration**
3. Name: `Local GPT` (or anything)
4. Supported account types: **Single tenant** (your org)
5. Redirect URI: **Public client/native** → `http://localhost:8400`
6. Click **Register**

### 2. Configure Permissions

1. Go to **API permissions** → Add a permission → Microsoft Graph
2. Add **Delegated permissions**: `Mail.Read`, `Mail.ReadBasic`, `User.Read`
3. Click **Grant admin consent** (or ask your admin)

### 3. Update .env

```
OUTLOOK_CLIENT_ID=<your-app-client-id>
OUTLOOK_TENANT_ID=<your-tenant-id>
```

### 4. Fetch Emails

```bash
# Via CLI
python ingest.py --outlook --days 30

# Or via the web UI sidebar → Outlook tab
```

First time, it opens a browser for Microsoft login. After that, tokens are cached locally.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│  RAG Engine   │────▶│  Ollama LLM  │
│   Web UI     │     │  (LlamaIndex) │     │ (Llama 3.2)  │
└─────────────┘     └──────┬───────┘     └──────────────┘
                           │
                    ┌──────▼───────┐
                    │   ChromaDB    │
                    │ Vector Store  │
                    └──────▲───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼────┐ ┌────▼─────┐ ┌───▼────┐
        │ Documents │ │  Outlook  │ │ Upload │
        │  (batch)  │ │  Emails   │ │  (UI)  │
        └──────────┘ └──────────┘ └────────┘
```

- **Ollama** runs Llama 3.2 locally on Apple Silicon
- **nomic-embed-text** generates embeddings locally (also via Ollama)
- **ChromaDB** stores vectors on disk — no external database needed
- **LlamaIndex** handles chunking, retrieval, and query orchestration

## Tips

- **More RAM = more context**: Llama 3.2 (3B) works great on 8GB+. For longer documents, try `llama3.2:latest` or `llama3.1:8b` if you have 16GB+.
- **Re-ingest is safe**: Duplicate documents are detected by content hash and skipped.
- **Clear and rebuild**: Use `python ingest.py --clear /path` if you restructured your docs.
