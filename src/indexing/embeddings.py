"""Embedding generation and ChromaDB storage module."""
import os
import shutil
from typing import List, Optional, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import settings


def _filter_complex_metadata(doc: Document) -> Document:
    """
    Filter out complex metadata types (lists, dicts) that ChromaDB doesn't support.
    Only keeps str, int, float, bool values.
    
    Args:
        doc: LangChain Document object
        
    Returns:
        Document with filtered metadata
    """
    filtered_metadata = {}
    for key, value in doc.metadata.items():
        # ChromaDB only supports: str, int, float, bool
        if isinstance(value, (str, int, float, bool)):
            filtered_metadata[key] = value
        # Convert lists to comma-separated strings
        elif isinstance(value, list):
            if value:  # Only convert non-empty lists
                filtered_metadata[key] = ", ".join(str(item) for item in value)
        # Skip dicts and other complex types
        else:
            continue
    
    return Document(
        page_content=doc.page_content,
        metadata=filtered_metadata
    )


class EmbeddingStore:
    """Handles embedding generation and storage in ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "products",
        clear_existing: bool = False
    ):
        """
        Initialize the embedding store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            clear_existing: Whether to clear existing index before initializing
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = settings.openai_model
        
        # Ensure parent directory exists
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear existing collection if requested
        # Note: For multi-collection setup, we clear the entire directory to ensure clean rebuild
        # This is safe since we always do full rebuilds of all collections together
        if clear_existing and self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            print(f"  ✓ Cleared existing vector store at {self.persist_directory}")
        
        # Initialize OpenAI embedding model
        # Support OpenRouter by using OPENAI_API_BASE if provided
        # When OPENAI_API_BASE is set (e.g., https://openrouter.ai/api/v1), 
        # the codebase uses OpenAI through OpenRouter
        embedding_kwargs = {
            "model": settings.openai_model,
            "openai_api_key": settings.openai_api_key
        }
        
        # If OPENAI_API_BASE is set, use it (for OpenRouter or other providers)
        if settings.openai_api_base:
            embedding_kwargs["openai_api_base"] = settings.openai_api_base
        
        self.embeddings = OpenAIEmbeddings(**embedding_kwargs)
        
        # Initialize ChromaDB (always create new since we clear on rebuild)
        self.vectorstore: Optional[Chroma] = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize ChromaDB vectorstore."""
        # Create new vectorstore (since we always rebuild)
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to ChromaDB with embeddings.
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process per batch
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Process in batches for better performance with large datasets
        all_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            # Filter complex metadata (lists, dicts, etc.) to ensure ChromaDB compatibility
            filtered_batch = [_filter_complex_metadata(doc) for doc in batch]
            ids = self.vectorstore.add_documents(filtered_batch)
            all_ids.extend(ids)
        
        # Chroma automatically persists when persist_directory is set during initialization
        # No need to call persist() explicitly
        
        return all_ids
    
    def update_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ):
        """
        Update existing documents in ChromaDB.
        
        Args:
            documents: List of Document objects to update
            ids: Optional list of document IDs (if None, will be generated)
        """
        if not documents:
            return
        
        if ids:
            self.vectorstore.update_documents(documents, ids=ids)
        else:
            self.vectorstore.update_documents(documents)
        
        # Chroma automatically persists when persist_directory is set
    
    def delete_documents(self, ids: List[str]):
        """
        Delete documents from ChromaDB.
        
        Args:
            ids: List of document IDs to delete
        """
        if ids:
            self.vectorstore.delete(ids=ids)
            # Chroma automatically persists when persist_directory is set
    
    def delete_by_metadata(
        self,
        filter_dict: dict
    ):
        """
        Delete documents by metadata filter.
        
        Args:
            filter_dict: Dictionary of metadata filters (e.g., {"product_id": 1})
        """
        # ChromaDB uses where filters
        self.vectorstore.delete(where=filter_dict)
        # Chroma automatically persists when persist_directory is set
    
    def get_vectorstore(self) -> Chroma:
        """
        Get the ChromaDB vectorstore instance.
        
        Returns:
            ChromaDB vectorstore
        """
        return self.vectorstore
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Perform similarity search using cosine similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar Document objects
        """
        if filter_dict:
            return self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        if filter_dict:
            return self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the ChromaDB collection.
        
        Returns:
            Dictionary with collection statistics
        """
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "embedding_model": self.embedding_model_name,
            "similarity_metric": "cosine"
        }

