#!/usr/bin/env python3
"""
CLI tool for batch document ingestion.

Usage:
    # Ingest a single folder
    python ingest.py /path/to/documents

    # Ingest multiple folders and files
    python ingest.py /path/to/docs /path/to/file.pdf ~/Desktop/notes

    # Ingest the default ./documents folder
    python ingest.py

    # Ingest Outlook emails (last 30 days)
    python ingest.py --outlook --days 30

    # Ingest everything (documents + outlook)
    python ingest.py /path/to/docs --outlook

    # Clear the database and re-ingest
    python ingest.py --clear /path/to/docs

    # Remove chunks whose filenames contain text (e.g. bad exports), then exit
    python ingest.py --purge-filename-substring "Music Software Directory" --purge-only

    # Watch a folder for new files
    python ingest.py --watch /path/to/docs
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

from config import DOCUMENTS_DIR
from document_loader import load_documents, scan_directory
from vector_store import LocalVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BATCH_SIZE = 10


def ingest_paths(store: LocalVectorStore, paths: list[Path]) -> int:
    """Ingest documents from given paths into the vector store, saving every BATCH_SIZE docs."""
    total_files = 0
    for p in paths:
        if p.is_dir():
            total_files += len(scan_directory(p))
        elif p.is_file():
            total_files += 1

    if total_files == 0:
        logger.warning("No supported files found.")
        return 0

    logger.info(f"Found {total_files} files to process (saving every {BATCH_SIZE} docs)...")

    total_added = 0
    batch = []
    processed = 0

    for doc in tqdm(load_documents(paths), total=total_files, desc="Ingesting"):
        batch.append(doc)
        processed += 1

        if len(batch) >= BATCH_SIZE:
            try:
                added = store.add_documents(batch)
                total_added += added
                logger.info(
                    f"Batch saved: +{added} docs | "
                    f"Total: {total_added} added, {processed}/{total_files} processed"
                )
            except Exception as e:
                logger.error(f"Batch save failed: {e} — skipping {len(batch)} docs")
            batch = []

    if batch:
        try:
            added = store.add_documents(batch)
            total_added += added
            logger.info(f"Final batch saved: +{added} docs")
        except Exception as e:
            logger.error(f"Final batch save failed: {e}")

    logger.info(f"Done! Added {total_added} new documents from {processed} files.")
    return total_added


def ingest_outlook(store: LocalVectorStore, days: int, max_emails: int) -> int:
    """Fetch and ingest Outlook emails."""
    from outlook_client import OutlookClient

    client = OutlookClient()
    logger.info(f"Fetching emails from last {days} days...")
    emails = client.fetch_emails(days_back=days, max_emails=max_emails)

    if not emails:
        logger.warning("No emails fetched.")
        return 0

    logger.info(f"Fetched {len(emails)} emails. Indexing...")
    added = store.add_documents(emails)
    logger.info(f"Done! Added {added} email documents.")
    return added


def watch_directory(store: LocalVectorStore, directory: Path):
    """Watch a directory for new files and auto-ingest them."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    from config import SUPPORTED_EXTENSIONS

    class IngestHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                logger.info(f"New file detected: {path.name}")
                time.sleep(1)  # wait for file write to complete
                ingest_paths(store, [path])

    observer = Observer()
    observer.schedule(IngestHandler(), str(directory), recursive=True)
    observer.start()
    logger.info(f"Watching {directory} for new files... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def main():
    parser = argparse.ArgumentParser(
        description="Batch ingest documents into Local GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to ingest (default: ./documents)",
    )
    parser.add_argument(
        "--outlook",
        action="store_true",
        help="Also fetch and ingest Outlook 365 emails",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days back to fetch emails (default: 30)",
    )
    parser.add_argument(
        "--max-emails",
        type=int,
        default=200,
        help="Maximum emails to fetch (default: 200)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the vector store before ingesting",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch directories for new files after initial ingest",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show vector store stats and exit",
    )
    parser.add_argument(
        "--purge-filename-substring",
        action="append",
        dest="purge_substrings",
        metavar="TEXT",
        help=(
            "Delete indexed chunks whose filename contains TEXT (case-insensitive). "
            "Repeat the flag for multiple patterns."
        ),
    )
    parser.add_argument(
        "--purge-only",
        action="store_true",
        help="With --purge-filename-substring, exit after purge (no folder ingest).",
    )

    args = parser.parse_args()
    store = LocalVectorStore()

    if args.stats:
        stats = store.get_stats()
        print(f"\nVector Store Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.purge_substrings:
        removed = store.delete_chunks_by_filename_substrings(args.purge_substrings)
        print(f"\nPurged {removed} chunk(s) matching filename substrings.")
        if args.purge_only:
            stats = store.get_stats()
            print(f"Remaining chunks: {stats['total_chunks']}")
            print("Restart Streamlit so the chat UI reloads the vector index.")
            return

    if args.clear:
        logger.info("Clearing vector store...")
        store.clear()
        store = LocalVectorStore()

    paths = args.paths if args.paths else [DOCUMENTS_DIR]

    for p in paths:
        if p.is_dir() or p.is_file():
            continue
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {p}")

    total_added = ingest_paths(store, paths)

    if args.outlook:
        total_added += ingest_outlook(store, args.days, args.max_emails)

    stats = store.get_stats()
    print(f"\nSummary: Added {total_added} documents. Total chunks: {stats['total_chunks']}")

    if args.watch:
        dirs = [p for p in paths if p.is_dir()]
        if dirs:
            watch_directory(store, dirs[0])


if __name__ == "__main__":
    main()
