"""Phase 7 evaluation package.

Golden dataset, metrics, judge and runner for the permission-aware RAG system.
Nothing here re-implements retrieval, ranking or authorization: every case is
executed through the Phase 5 service, which is the same code path the chat UI
uses.
"""
from __future__ import annotations
