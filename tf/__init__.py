"""
TF - Text Semantic Retrieval System SDK

A high-performance text semantic retrieval system combining Python and Rust.

Quick Start:
    >>> from tf import DocumentStore
    >>> store = DocumentStore()
    >>> store.add("doc1", "Content here", title="Title", summary="Summary")
    >>> results = store.search("query text", k=5)
"""

from .embeddings import TextEmbedder
from .vector_store import VectorStoreWrapper
from .sdk import DocumentStore, SDK

__all__ = ['TextEmbedder', 'VectorStoreWrapper', 'DocumentStore', 'SDK']
__version__ = '0.1.0'
