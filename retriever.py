# retriever.py
"""
Retrieval Layer
Loads FAISS index and performs semantic search.
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer


class DocumentRetriever:
    def __init__(self, index_dir: str = "./index", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize retriever with FAISS index.
        
        Args:
            index_dir: Directory containing FAISS index and chunks
            model_name: sentence-transformers model name (must match ingestion)
        """
        self.index_dir = Path(index_dir)
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index and chunks from disk."""
        index_path = self.index_dir / "faiss.index"
        chunks_path = self.index_dir / "chunks.pkl"
        
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for query.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (chunk_text, distance) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Return chunks with distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve and format context as plain text.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, (chunk, distance) in enumerate(results, 1):
            context_parts.append(f"[Context {i}]\n{chunk}\n")
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    # Example usage
    retriever = DocumentRetriever(index_dir="./index")
    
    query = "What is machine learning?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nQuery: {query}\n")
    for i, (chunk, distance) in enumerate(results, 1):
        print(f"Result {i} (distance: {distance:.4f}):")
        print(f"{chunk[:200]}...\n")