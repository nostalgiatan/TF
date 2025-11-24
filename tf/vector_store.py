"""
Vector store wrapper that integrates Python embeddings with Rust vector storage.
"""

from typing import List, Dict, Optional, Any
from .embeddings import TextEmbedder


class VectorStoreWrapper:
    """
    High-level wrapper that combines TextEmbedder with the Rust VectorStore.
    
    This class provides a convenient interface for adding documents,
    searching by text, and managing the vector database.
    """
    
    def __init__(self, embedder: TextEmbedder = None, rust_store=None):
        """
        Initialize the vector store wrapper.
        
        Args:
            embedder: TextEmbedder instance (created if None)
            rust_store: Rust VectorStore instance (created if None)
        """
        # Initialize embedder
        if embedder is None:
            self.embedder = TextEmbedder()
        else:
            self.embedder = embedder
        
        # Initialize Rust store
        if rust_store is None:
            try:
                from tf_rust import VectorStore
                self.store = VectorStore(self.embedder.get_dimension())
            except ImportError as e:
                raise ImportError(
                    "Failed to import tf_rust module. "
                    "Please build the Rust extension first using 'maturin develop' or 'maturin build'."
                ) from e
        else:
            self.store = rust_store
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        title: str = "",
        url: str = ""
    ) -> None:
        """
        Add a document to the vector store.
        
        The content will be automatically converted to a vector embedding.
        
        Args:
            doc_id: Unique identifier for the document
            content: Document content (will be embedded)
            title: Document title (optional)
            url: Document URL (optional)
        """
        # Generate embedding for content
        embedding = self.embedder.encode(content)
        
        # Ensure embedding is a flat list of floats
        if isinstance(embedding[0], list):
            # If encode returned a list of embeddings, take the first one
            embedding = embedding[0]
        
        # Add to Rust store
        self.store.set(doc_id, embedding, title, url, content)
    
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Add multiple documents at once.
        
        Args:
            documents: List of document dictionaries with keys:
                      - id: Document ID (required)
                      - content: Document content (required)
                      - title: Document title (optional)
                      - url: Document URL (optional)
        """
        for doc in documents:
            doc_id = doc.get('id')
            content = doc.get('content')
            
            if not doc_id or not content:
                raise ValueError("Each document must have 'id' and 'content' fields")
            
            title = doc.get('title', '')
            url = doc.get('url', '')
            
            self.add_document(doc_id, content, title, url)
    
    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query text.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of result dictionaries with keys: id, score, title, url, content
        """
        # Generate embedding for query
        query_embedding = self.embedder.encode(query)
        
        # Ensure embedding is a flat list of floats
        if isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
        
        # Search in Rust store
        results = self.store.search(query_embedding, k)
        
        return results
    
    def search_by_embedding(
        self,
        embedding: List[float],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using a pre-computed embedding.
        
        Args:
            embedding: Query embedding (list of floats)
            k: Number of results to return
            
        Returns:
            List of result dictionaries with keys: id, score, title, url, content
        """
        return self.store.search(embedding, k)
    
    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the store.
        
        Args:
            doc_id: Document ID to remove
        """
        self.store.rm(doc_id)
    
    def get_metadata(self, doc_id: str) -> Optional[Dict[str, str]]:
        """
        Get metadata for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary with title, url, and content, or None if not found
        """
        return self.store.get_metadata(doc_id)
    
    def __len__(self) -> int:
        """Get the number of documents in the store."""
        return self.store.len()
    
    def is_empty(self) -> bool:
        """Check if the store is empty."""
        return self.store.is_empty()
