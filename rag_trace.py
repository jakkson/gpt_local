"""
LlamaIndex callback handler that turns retrieval / LLM events into short status lines
for Streamlit (or any sink).
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks import CallbackManager


def build_trace_callback_manager(sink: Callable[[str], None]) -> CallbackManager:
    return CallbackManager([StreamlitRagTraceHandler(sink)])


class StreamlitRagTraceHandler(BaseCallbackHandler):
    """Maps high-level LlamaIndex events to user-visible trace lines."""

    def __init__(self, sink: Callable[[str], None]) -> None:
        super().__init__(
            event_starts_to_ignore=[
                CBEventType.CHUNKING,
                CBEventType.NODE_PARSING,
                CBEventType.TEMPLATING,
            ],
            event_ends_to_ignore=[
                CBEventType.CHUNKING,
                CBEventType.NODE_PARSING,
                CBEventType.TEMPLATING,
            ],
        )
        self._sink = sink

    def _log(self, line: str) -> None:
        try:
            self._sink(line)
        except Exception:
            pass

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type in self.event_starts_to_ignore:
            return event_id or str(uuid.uuid4())
        if event_type == CBEventType.QUERY:
            q = (payload or {}).get(EventPayload.QUERY_STR, "")
            if q:
                short = q if len(q) <= 120 else q[:117] + "…"
                self._log(f"**Query:** {short}")
            else:
                self._log("Running search over the index…")
        elif event_type == CBEventType.RETRIEVE:
            self._log("Retrieving document chunks (embeddings + keywords)…")
        elif event_type == CBEventType.SYNTHESIZE:
            self._log("Composing an answer from the retrieved passages…")
        elif event_type == CBEventType.LLM:
            self._log("Calling the local LLM (70B can take a minute)…")
        elif event_type == CBEventType.EMBEDDING:
            self._log("Embedding query for vector search…")
        return event_id or str(uuid.uuid4())

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type in self.event_ends_to_ignore:
            return
        payload = payload or {}
        if event_type == CBEventType.RETRIEVE:
            nodes = payload.get(EventPayload.NODES)
            if nodes is not None:
                self._log(f"Retrieved **{len(nodes)}** chunk(s) for context.")
        elif event_type == CBEventType.LLM:
            self._log("LLM response received.")
        elif event_type == CBEventType.SYNTHESIZE:
            self._log("Answer synthesis complete.")
        elif event_type == CBEventType.EMBEDDING:
            self._log("Embedding done.")

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._log("---")

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._log("---")
