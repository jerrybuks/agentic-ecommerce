"""Storage utility functions for indexing operations."""
from typing import List, Dict
from langchain_core.documents import Document
from src.indexing.embeddings import EmbeddingStore


def store_products_in_vectorstore(
    product_chunks: List[Document],
    batch_size: int = 100,
    clear_existing: bool = True
) -> Dict:
    """
    Store product chunks in the products vector store.
    
    Args:
        product_chunks: List of product chunk documents to store
        batch_size: Number of chunks to process per batch
        clear_existing: Whether to clear existing products collection
        
    Returns:
        Dictionary with final statistics
    """
    if not product_chunks:
        print("⚠ No product chunks to store.")
        return {}
    
    print("\nStoring products in vector store...")
    print(f"  Collection: products")
    
    # Initialize products embedding store
    # clear_existing=True will clear the entire vector_store directory
    # This is fine since we rebuild all collections together
    products_store = EmbeddingStore(
        persist_directory="data/vector_store",
        collection_name="products",
        clear_existing=clear_existing
    )
    
    # Generate embeddings and store in ChromaDB
    print(f"  Generating embeddings and storing {len(product_chunks)} product chunks...")
    document_ids = products_store.add_documents(
        product_chunks,
        batch_size=batch_size
    )
    
    print(f"  ✓ Stored {len(document_ids)} product chunks in 'products' collection")
    
    # Get final statistics
    final_stats = products_store.get_collection_stats()
    return final_stats


def store_handbook_in_vectorstore(
    handbook_chunks: List[Document],
    batch_size: int = 100,
    clear_existing: bool = True
) -> Dict:
    """
    Store handbook chunks in the general_handbook vector store.
    
    Args:
        handbook_chunks: List of handbook chunk documents to store
        batch_size: Number of chunks to process per batch
        clear_existing: Whether to clear existing handbook collection
        
    Returns:
        Dictionary with final statistics
    """
    if not handbook_chunks:
        print("⚠ No handbook chunks to store.")
        return {}
    
    print("\nStoring handbook in vector store...")
    print(f"  Collection: general_handbook")
    
    # Initialize handbook embedding store
    # clear_existing=False here since products collection already cleared the directory
    # Both collections are in the same persist_directory, so we only clear once
    handbook_store = EmbeddingStore(
        persist_directory="data/vector_store",
        collection_name="general_handbook",
        clear_existing=False  # Already cleared by products collection
    )
    
    # Generate embeddings and store in ChromaDB
    print(f"  Generating embeddings and storing {len(handbook_chunks)} handbook chunks...")
    document_ids = handbook_store.add_documents(
        handbook_chunks,
        batch_size=batch_size
    )
    
    print(f"  ✓ Stored {len(document_ids)} handbook chunks in 'general_handbook' collection")
    
    # Get final statistics
    final_stats = handbook_store.get_collection_stats()
    return final_stats


def update_index(product_ids: list = None):
    """
    Update the index for specific products or all products.
    
    Args:
        product_ids: List of product IDs to update (None for all)
    """
    print("=" * 60)
    print("Updating Product Index")
    print("=" * 60)
    
    # This would re-parse, re-chunk, and update specific products
    # Implementation can be added as needed
    print("Update functionality can be implemented as needed")
