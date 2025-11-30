"""
FAISS-based vector store for entity embeddings.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json

try:
    import faiss
except ImportError:
    faiss = None

from ..config import EMBEDDING_DIMENSION


class VectorStore:
    """FAISS-based vector store for similarity search."""

    def __init__(self, store_path: str, dimension: int = EMBEDDING_DIMENSION):
        self.store_path = Path(store_path)
        self.dimension = dimension
        self.index_path = self.store_path / "embeddings.faiss"
        self.mapping_path = self.store_path / "id_mapping.json"

        self.store_path.mkdir(parents=True, exist_ok=True)

        # ID to index mapping
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}

        # Initialize or load index
        if faiss is not None:
            self._init_faiss()
        else:
            self.index = None
            self._fallback_vectors: Dict[str, np.ndarray] = {}

    def _init_faiss(self):
        """Initialize FAISS index."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self._load_mapping()
        else:
            # Use L2 distance (Euclidean) - suitable for normalized embeddings
            self.index = faiss.IndexFlatL2(self.dimension)

    def _load_mapping(self):
        """Load ID mapping from disk."""
        if self.mapping_path.exists():
            with open(self.mapping_path, 'r') as f:
                data = json.load(f)
                self.id_to_idx = data.get("id_to_idx", {})
                self.idx_to_id = {int(k): v for k, v in data.get("idx_to_id", {}).items()}

    def _save_mapping(self):
        """Save ID mapping to disk."""
        with open(self.mapping_path, 'w') as f:
            json.dump({
                "id_to_idx": self.id_to_idx,
                "idx_to_id": {str(k): v for k, v in self.idx_to_id.items()}
            }, f)

    def add(self, entity_id: str, embedding: np.ndarray):
        """Add an embedding for an entity."""
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embedding.shape[0]}")

        # Normalize for cosine similarity
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if faiss is not None and self.index is not None:
            idx = self.index.ntotal
            self.index.add(embedding.reshape(1, -1))
            self.id_to_idx[entity_id] = idx
            self.idx_to_id[idx] = entity_id
        else:
            # Fallback: store in dictionary
            self._fallback_vectors[entity_id] = embedding

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar entities.

        Returns list of (entity_id, distance) tuples, sorted by similarity.
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {query_embedding.shape[0]}")

        # Normalize query
        query_embedding = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        if faiss is not None and self.index is not None:
            if self.index.ntotal == 0:
                return []

            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx in self.idx_to_id:
                    # Convert L2 distance to similarity score (lower is better for L2)
                    similarity = 1 / (1 + dist)
                    results.append((self.idx_to_id[idx], similarity))
            return results
        else:
            # Fallback: brute force search
            if not self._fallback_vectors:
                return []

            results = []
            for entity_id, vec in self._fallback_vectors.items():
                # Cosine similarity (vectors are normalized)
                similarity = float(np.dot(query_embedding, vec))
                results.append((entity_id, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def remove(self, entity_id: str):
        """Remove an entity's embedding (marks for rebuild)."""
        # FAISS doesn't support efficient deletion, so we mark for rebuild
        if entity_id in self.id_to_idx:
            del self.id_to_idx[entity_id]
        if entity_id in self._fallback_vectors:
            del self._fallback_vectors[entity_id]

    def save(self):
        """Save index to disk."""
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self._save_mapping()

    def load(self):
        """Load index from disk."""
        if faiss is not None:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self._load_mapping()

    def get_count(self) -> int:
        """Get number of stored embeddings."""
        if faiss is not None and self.index is not None:
            return self.index.ntotal
        return len(self._fallback_vectors)

    def has_entity(self, entity_id: str) -> bool:
        """Check if entity has an embedding."""
        return entity_id in self.id_to_idx or entity_id in self._fallback_vectors


class EmbeddingGenerator:
    """Generate embeddings using Gemini API."""

    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.genai = genai
        # Use text-embedding model for embeddings
        self.model_name = "models/text-embedding-004"

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        try:
            result = self.genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    def generate_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.generate(text))
        return embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query (different task type for retrieval)."""
        try:
            result = self.genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
