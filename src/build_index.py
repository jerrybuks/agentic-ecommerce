"""Main script to build the product index."""
import sys
from pathlib import Path
from typing import List
from langchain_core.documents import Document

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.parsing import ProductParser, HandbookParser
from src.indexing.chunking import ProductChunker, MarkdownChunker
from src.utils.storage import store_products_in_vectorstore, store_handbook_in_vectorstore


def index_products(
    batch_size: int = 100,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    is_active_only: bool = True
) -> List[Document]:
    """
    Index products from the database.
    
    Args:
        batch_size: Number of products to process per batch
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        is_active_only: Only index active products
        
    Returns:
        List of chunked Document objects
    """
    print("\n[1/3] Indexing Products")
    print("-" * 60)
    
    # Parse products from database
    print("Parsing products from database...")
    with ProductParser() as parser:
        parsed_products = parser.parse_all_products(
            batch_size=batch_size,
            is_active_only=is_active_only
        )
        print(f"✓ Parsed {len(parsed_products)} products")
    
    if not parsed_products:
        print("⚠ No products found to index.")
        return []
    
    # Convert to LangChain documents
    product_documents = parser.to_langchain_documents(parsed_products)
    print(f"✓ Converted to {len(product_documents)} LangChain documents")
    
    # Chunk product documents
    print("\nChunking product documents...")
    product_chunker = ProductChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    product_chunks = product_chunker.chunk_documents(product_documents)
    
    # Get chunking statistics
    stats = product_chunker.get_chunk_statistics(product_chunks)
    print(f"✓ Created {stats['total_chunks']} product chunks")
    print(f"  - Average chunk size: {stats['avg_chunk_size']:.0f} characters")
    print(f"  - Min chunk size: {stats['min_chunk_size']} characters")
    print(f"  - Max chunk size: {stats['max_chunk_size']} characters")
    
    # Save product chunks to JSONL file
    print("\nSaving product chunks to JSONL file...")
    chunks_saved = product_chunker.save_chunks_to_jsonl(
        product_chunks,
        file_path="src/data/jsonl/product_chunks.jsonl"
    )
    print(f"✓ Saved {chunks_saved} chunks to src/data/jsonl/product_chunks.jsonl")
    
    return product_chunks


def index_handbook(
    handbook_path: str = "src/data/handbooks/general_handbook.md",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Index handbook markdown file.
    
    Args:
        handbook_path: Path to the handbook markdown file
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    print("\n[2/3] Indexing Handbook")
    print("-" * 60)
    
    try:
        # Parse handbook
        print(f"Loading handbook from {handbook_path}...")
        handbook_parser = HandbookParser(handbook_path)
        handbook_content = handbook_parser.load_handbook()
        parsed_handbook = handbook_parser.parse_handbook(handbook_content)
        print(f"✓ Loaded handbook: {parsed_handbook['metadata']['handbook_name']}")
        
        # Chunk handbook using markdown chunker
        print("\nChunking handbook by headers...")
        markdown_chunker = MarkdownChunker()
        handbook_chunks = markdown_chunker.chunk_markdown(
            handbook_content,
            metadata=parsed_handbook["metadata"],
            preserve_metadata=True
        )
        print(f"✓ Created {len(handbook_chunks)} handbook chunks")
        
        # Save handbook chunks to JSONL file
        print("\nSaving handbook chunks to JSONL file...")
        jsonl_chunker = ProductChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks_saved = jsonl_chunker.save_chunks_to_jsonl(
            handbook_chunks,
            file_path="src/data/jsonl/handbook_chunks.jsonl"
        )
        print(f"✓ Saved {chunks_saved} chunks to src/data/jsonl/handbook_chunks.jsonl")
        
        return handbook_chunks
        
    except FileNotFoundError as e:
        print(f"⚠ Handbook not found: {e}")
        print("  Continuing without handbook...")
        return []
    except Exception as e:
        print(f"⚠ Error processing handbook: {e}")
        print("  Continuing without handbook...")
        return []


def build_index(
    batch_size: int = 100,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    is_active_only: bool = True
):
    """
    Build the complete index by running the full indexing pipeline.
    Always performs a full rebuild of the ChromaDB index.
    
    Args:
        batch_size: Number of products to process per batch
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        is_active_only: Only index active products
    """
    print("=" * 60)
    print("Building Product & Handbook Index")
    print("=" * 60)
    
    # Index products
    product_chunks = index_products(
        batch_size=batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_active_only=is_active_only
    )
    
    # Index handbook
    handbook_chunks = index_handbook(
        handbook_path="src/data/handbooks/general_handbook.md",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not product_chunks and not handbook_chunks:
        print("\n⚠ No documents to index. Exiting.")
        return
    
    # Store products and handbook in separate vector stores
    print("\n[3/3] Storing in Vector Stores")
    print("-" * 60)
    print("Initializing vector stores...")
    print("  Performing full rebuild (clearing existing collections)...")
    
    products_stats = {}
    handbook_stats = {}
    
    # Store products in 'products' collection
    if product_chunks:
        products_stats = store_products_in_vectorstore(
            product_chunks,
            batch_size=batch_size,
            clear_existing=True
        )
    
    # Store handbook in 'general_handbook' collection
    if handbook_chunks:
        handbook_stats = store_handbook_in_vectorstore(
            handbook_chunks,
            batch_size=batch_size,
            clear_existing=True
        )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Index Build Complete!")
    print("=" * 60)
    
    if products_stats:
        print(f"\nProducts Collection:")
        print(f"  Collection Name: {products_stats.get('collection_name', 'N/A')}")
        print(f"  Total Documents: {products_stats.get('total_documents', 0)}")
        print(f"  Embedding Model: {products_stats.get('embedding_model', 'N/A')}")
        print(f"  Similarity Metric: {products_stats.get('similarity_metric', 'N/A')}")
    
    if handbook_stats:
        print(f"\nHandbook Collection:")
        print(f"  Collection Name: {handbook_stats.get('collection_name', 'N/A')}")
        print(f"  Total Documents: {handbook_stats.get('total_documents', 0)}")
        print(f"  Embedding Model: {handbook_stats.get('embedding_model', 'N/A')}")
        print(f"  Similarity Metric: {handbook_stats.get('similarity_metric', 'N/A')}")
    
    print(f"\nVector Store Location: src/data/vector_store")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build product index for semantic search")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of products to process per batch (default: 100)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)"
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive products in index"
    )
    
    args = parser.parse_args()
    
    build_index(
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        is_active_only=not args.include_inactive
    )
