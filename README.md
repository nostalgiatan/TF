# TF - Text Semantic Retrieval System

A high-performance text semantic retrieval system that combines Python (for embeddings) and Rust (for vector storage and search).

## Features

- **Semantic Search**: Use the Qwen3-Embedding-0.6B model to convert text into vector embeddings
- **Fast Vector Storage**: Leverage Rust's Voyager library for efficient vector storage and similarity search
- **Metadata Support**: Store and retrieve metadata (title, url, content) alongside vectors
- **Python-Rust Integration**: Seamless integration using PyO3 for optimal performance

## Architecture

```
┌─────────────────────────────────────────┐
│           Python Layer (tf/)            │
│  - TextEmbedder (Qwen3-Embedding)       │
│  - VectorStoreWrapper (High-level API)  │
└──────────────────┬──────────────────────┘
                   │ PyO3 bindings
┌──────────────────▼──────────────────────┐
│           Rust Layer (src/)             │
│  - VectorStore (Voyager integration)    │
│  - Operations: set, search, rm          │
│  - Metadata management                  │
└─────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Rust (latest stable version)
- Cargo

### Build Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/nostalgiatan/TF.git
   cd TF
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   pip install maturin
   ```

3. **Build the Rust extension**
   ```bash
   maturin develop --release
   ```

## Usage

### Basic Example

```python
from tf import TextEmbedder, VectorStoreWrapper

# Initialize
embedder = TextEmbedder()
store = VectorStoreWrapper(embedder)

# Add documents
store.add_document(
    doc_id="doc1",
    title="Python Programming",
    url="https://example.com/python",
    content="Python is a high-level programming language..."
)

# Search
results = store.search("What is Python?", k=5)
for result in results:
    print(f"{result['title']}: {result['score']}")

# Remove document
store.remove_document("doc1")
```

### Running the Example

```bash
python example.py
```

## API Reference

### Python API

#### TextEmbedder

```python
embedder = TextEmbedder(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    device=None  # Auto-detect GPU/CPU
)

# Encode text to embeddings
embedding = embedder.encode("Your text here")
embeddings = embedder.encode(["Text 1", "Text 2"])

# Get embedding dimension
dim = embedder.get_dimension()
```

#### VectorStoreWrapper

```python
store = VectorStoreWrapper(embedder)

# Add single document
store.add_document(doc_id, content, title="", url="")

# Add multiple documents
store.add_documents([
    {"id": "1", "content": "...", "title": "...", "url": "..."},
    ...
])

# Search by text
results = store.search(query, k=5)

# Search by embedding
results = store.search_by_embedding(embedding, k=5)

# Remove document
store.remove_document(doc_id)

# Get metadata
metadata = store.get_metadata(doc_id)

# Get count
count = len(store)
is_empty = store.is_empty()
```

### Rust API

The Rust VectorStore is exposed to Python via PyO3:

```python
from tf_rust import VectorStore

store = VectorStore(dimension=768)

# Set (add/update) vector with metadata
store.set(id, vector, title, url, content)

# Search for similar vectors
results = store.search(vector, k=5)
# Returns: [{"id": "...", "score": 0.95, "title": "...", ...}]

# Remove vector
store.rm(id)

# Get metadata
metadata = store.get_metadata(id)

# Utilities
count = store.len()
is_empty = store.is_empty()
```

## Project Structure

```
TF/
├── src/
│   └── lib.rs              # Rust implementation (VectorStore)
├── tf/
│   ├── __init__.py         # Python package init
│   ├── embeddings.py       # TextEmbedder implementation
│   └── vector_store.py     # VectorStoreWrapper implementation
├── Cargo.toml              # Rust dependencies
├── pyproject.toml          # Python package configuration
├── requirements.txt        # Python dependencies
├── example.py              # Usage example
└── README.md               # This file
```

## Technical Details

### Embeddings

- Model: Qwen3/Qwen3-Embedding-0.6B
- Pooling: Mean pooling over last hidden states
- Normalization: L2 normalization
- Dimension: Determined by model (typically 768)

### Vector Storage

- Backend: Voyager (high-performance vector search)
- Metric: Cosine similarity
- Index: HNSW (Hierarchical Navigable Small World)

### Metadata

Each vector is associated with:
- `id`: Unique identifier
- `title`: Document title
- `url`: Document URL
- `content`: Full document content

## Dependencies

### Rust Dependencies

- `pyo3`: Python bindings
- `voyager`: Vector search engine
- `serde`: Serialization
- `serde_json`: JSON support

### Python Dependencies

- `torch`: PyTorch for model inference
- `transformers`: Hugging Face transformers library
- `numpy`: Numerical operations

## Performance Tips

1. **Batch Processing**: Use `add_documents()` instead of multiple `add_document()` calls
2. **GPU Acceleration**: Ensure CUDA is available for faster embeddings
3. **Dimension**: Smaller embedding dimensions = faster search, but less accuracy

## License

This project is open source. Please check the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

