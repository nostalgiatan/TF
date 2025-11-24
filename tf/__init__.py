"""
TF - Text Semantic Retrieval System

A text semantic retrieval system that combines Python (for embeddings) and Rust (for vector storage and search).
"""

from .embeddings import TextEmbedder
from .vector_store import VectorStoreWrapper

__all__ = ['TextEmbedder', 'VectorStoreWrapper']
__version__ = '0.1.0'
