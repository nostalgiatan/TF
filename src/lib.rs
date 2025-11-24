use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use vecstore::{Metadata, Query, VecStore};

/// Vector store that manages embeddings and metadata using VecStore
#[pyclass]
struct VectorStore {
    store: Arc<Mutex<VecStore>>,
    dimension: usize,
    temp_path: Option<PathBuf>,
}

#[pymethods]
impl VectorStore {
    /// Create a new VectorStore instance
    #[new]
    fn new(dimension: usize) -> PyResult<Self> {
        // Create a temporary directory for the vector store
        let temp_dir = std::env::temp_dir().join(format!("tf_vecstore_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create temp directory: {}",
                e
            ))
        })?;

        let store = VecStore::open(&temp_dir).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create vector store: {}",
                e
            ))
        })?;

        Ok(VectorStore {
            store: Arc::new(Mutex::new(store)),
            dimension,
            temp_path: Some(temp_dir),
        })
    }

    /// Set (add/update) a vector with its metadata
    ///
    /// Args:
    ///     id: Unique identifier for the document
    ///     vector: List of floats representing the embedding
    ///     title: Document title
    ///     url: Document URL
    ///     content: Document content
    fn set(
        &mut self,
        id: String,
        vector: Vec<f32>,
        title: String,
        url: String,
        content: String,
    ) -> PyResult<()> {
        if vector.len() != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector dimension mismatch. Expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        // Create metadata
        let mut metadata = Metadata {
            fields: HashMap::new(),
        };
        metadata.fields.insert("title".to_string(), json!(title));
        metadata.fields.insert("url".to_string(), json!(url));
        metadata
            .fields
            .insert("content".to_string(), json!(content));

        // Upsert vector with metadata
        self.store
            .lock()
            .unwrap()
            .upsert(id, vector, metadata)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to add vector: {}",
                    e
                ))
            })?;

        Ok(())
    }

    /// Search for similar vectors
    ///
    /// Args:
    ///     vector: Query vector (list of floats)
    ///     k: Number of results to return (default: 5)
    ///
    /// Returns:
    ///     List of dictionaries containing id, score, title, url, and content
    fn search(&self, py: Python, vector: Vec<f32>, k: Option<usize>) -> PyResult<Py<PyList>> {
        if vector.len() != self.dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Vector dimension mismatch. Expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        let k = k.unwrap_or(5);

        // Create query
        let query = Query {
            vector,
            k,
            filter: None,
        };

        // Execute query
        let results = self.store.lock().unwrap().query(query).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Search failed: {}", e))
        })?;

        // Convert results to Python list
        let result_list = PyList::empty(py);

        for result in results {
            let dict = PyDict::new(py);
            dict.set_item("id", &result.id)?;
            dict.set_item("score", result.score)?;

            // Extract metadata fields
            if let Some(title) = result.metadata.fields.get("title") {
                if let Some(title_str) = title.as_str() {
                    dict.set_item("title", title_str)?;
                }
            }
            if let Some(url) = result.metadata.fields.get("url") {
                if let Some(url_str) = url.as_str() {
                    dict.set_item("url", url_str)?;
                }
            }
            if let Some(content) = result.metadata.fields.get("content") {
                if let Some(content_str) = content.as_str() {
                    dict.set_item("content", content_str)?;
                }
            }

            result_list.append(dict)?;
        }

        Ok(result_list.into())
    }

    /// Remove a vector and its metadata
    ///
    /// Args:
    ///     id: Unique identifier of the document to remove
    fn rm(&mut self, id: String) -> PyResult<()> {
        self.store.lock().unwrap().delete(&id).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to remove vector: {}",
                e
            ))
        })?;

        Ok(())
    }

    /// Get the number of vectors in the store
    fn len(&self) -> PyResult<usize> {
        Ok(self.store.lock().unwrap().len())
    }

    /// Check if the store is empty
    fn is_empty(&self) -> PyResult<bool> {
        Ok(self.store.lock().unwrap().is_empty())
    }

    /// Get metadata for a specific document
    ///
    /// Args:
    ///     id: Document identifier
    ///
    /// Returns:
    ///     Dictionary containing title, url, and content, or None if not found
    fn get_metadata(&self, py: Python, id: String) -> PyResult<Py<PyAny>> {
        let store = self.store.lock().unwrap();
        let all_records = store.list_active();

        // Find the record with matching id
        for record in all_records {
            if record.id == id {
                let dict = PyDict::new(py);

                if let Some(title) = record.metadata.fields.get("title") {
                    if let Some(title_str) = title.as_str() {
                        dict.set_item("title", title_str)?;
                    }
                }
                if let Some(url) = record.metadata.fields.get("url") {
                    if let Some(url_str) = url.as_str() {
                        dict.set_item("url", url_str)?;
                    }
                }
                if let Some(content) = record.metadata.fields.get("content") {
                    if let Some(content_str) = content.as_str() {
                        dict.set_item("content", content_str)?;
                    }
                }

                return Ok(dict.into());
            }
        }

        Ok(py.None())
    }
}

impl Drop for VectorStore {
    fn drop(&mut self) {
        // Clean up temporary directory
        if let Some(ref path) = self.temp_path {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

/// PyO3 module definition
#[pymodule]
fn tf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorStore>()?;
    Ok(())
}
