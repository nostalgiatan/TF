"""
Example usage of the TF text semantic retrieval system.

This script demonstrates how to:
1. Initialize the vector store with embeddings
2. Add documents with metadata
3. Search for similar documents
4. Remove documents
"""

from tf import TextEmbedder, VectorStoreWrapper


def main():
    print("=== TF Text Semantic Retrieval System Example ===\n")
    
    # Initialize the embedder and vector store
    print("Step 1: Initializing embedder and vector store...")
    embedder = TextEmbedder()
    store = VectorStoreWrapper(embedder)
    print(f"✓ Vector store initialized (dimension: {embedder.get_dimension()})\n")
    
    # Add some example documents
    print("Step 2: Adding documents...")
    documents = [
        {
            "id": "doc1",
            "title": "Python Programming",
            "url": "https://example.com/python",
            "content": "Python is a high-level, interpreted programming language known for its simplicity and readability."
        },
        {
            "id": "doc2",
            "title": "Rust Programming",
            "url": "https://example.com/rust",
            "content": "Rust is a systems programming language that runs blazingly fast and prevents segfaults."
        },
        {
            "id": "doc3",
            "title": "JavaScript Basics",
            "url": "https://example.com/javascript",
            "content": "JavaScript is a versatile programming language primarily used for web development."
        },
        {
            "id": "doc4",
            "title": "Machine Learning",
            "url": "https://example.com/ml",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
        },
        {
            "id": "doc5",
            "title": "Vector Databases",
            "url": "https://example.com/vectordb",
            "content": "Vector databases are specialized databases for storing and searching high-dimensional vectors."
        }
    ]
    
    store.add_documents(documents)
    print(f"✓ Added {len(store)} documents\n")
    
    # Search for similar documents
    print("Step 3: Searching for documents...")
    queries = [
        "What is a programming language for systems development?",
        "Tell me about databases for embeddings",
        "How does AI learn from information?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = store.search(query, k=3)
        
        print("Results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} (score: {result['score']:.4f})")
            print(f"     URL: {result['url']}")
            print(f"     Content: {result['content'][:80]}...")
    
    # Get metadata for a specific document
    print("\n\nStep 4: Getting metadata for a document...")
    metadata = store.get_metadata("doc2")
    if metadata:
        print(f"Document 'doc2' metadata:")
        print(f"  Title: {metadata['title']}")
        print(f"  URL: {metadata['url']}")
        print(f"  Content: {metadata['content'][:80]}...")
    
    # Remove a document
    print("\n\nStep 5: Removing a document...")
    print(f"Documents before removal: {len(store)}")
    store.remove_document("doc3")
    print(f"Documents after removal: {len(store)}")
    print("✓ Document 'doc3' removed")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
