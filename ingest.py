# ingest.py
"""
Document Ingestion & Embeddings
Loads documents, chunks them, generates embeddings, and stores in FAISS.
"""

import os
from pathlib import Path
from typing import List
import pickle

from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2


class DocumentIngestion:
    def __init__(self, chunk_size: int = 600, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize document ingestion pipeline.
        
        Args:
            chunk_size: Approximate number of characters per chunk
            model_name: sentence-transformers model name
        """
        self.chunk_size = chunk_size
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.index = None
    
    def load_documents(self, data_dir: str) -> List[str]:
        """Load all PDF and text documents from directory."""
        documents = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load PDF files
        for pdf_file in data_path.glob("*.pdf"):
            print(f"Loading PDF: {pdf_file.name}")
            try:
                with open(pdf_file, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    documents.append(text)
            except Exception as e:
                print(f"Error loading {pdf_file.name}: {e}")
        
        # Load text files
        for txt_file in data_path.glob("*.txt"):
            print(f"Loading text: {txt_file.name}")
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
            except Exception as e:
                print(f"Error loading {txt_file.name}: {e}")
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Split documents into chunks."""
        chunks = []
        
        for doc in documents:
            # Simple character-based chunking with overlap
            overlap = self.chunk_size // 4
            start = 0
            
            while start < len(doc):
                end = start + self.chunk_size
                chunk = doc[start:end].strip()
                
                if chunk:
                    chunks.append(chunk)
                
                start = end - overlap
        
        print(f"Created {len(chunks)} chunks")
        self.chunks = chunks
        return chunks
    
    def generate_embeddings(self, chunks: List[str]):
        """Generate embeddings for all chunks."""
        print("Generating embeddings...")
        self.embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
    
    def build_faiss_index(self):
        """Build FAISS index from embeddings."""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Run generate_embeddings() first.")
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def save_index(self, index_dir: str = "./index"):
        """Save FAISS index and chunks to disk."""
        index_path = Path(index_dir)
        index_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path / "faiss.index"))
        
        # Save chunks
        with open(index_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved index to {index_dir}")
    
    def ingest(self, data_dir: str, index_dir: str = "./index"):
        """Complete ingestion pipeline."""
        documents = self.load_documents(data_dir)
        chunks = self.chunk_documents(documents)
        self.generate_embeddings(chunks)
        self.build_faiss_index()
        self.save_index(index_dir)
        print("Ingestion complete!")


if __name__ == "__main__":
    # Example usage
    ingestion = DocumentIngestion(chunk_size=600)
    ingestion.ingest(data_dir="./data", index_dir="./index")