"""Retrieval tools for agents using OpenAI function calling."""
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.indexing.embeddings import EmbeddingStore
from src.config import settings
from src.utils.similarity import filter_by_similarity_threshold


def get_handbook_retrieval_function(min_similarity: float = 0.75) -> Dict[str, Any]:
    """
    Get OpenAI function definition for handbook retrieval.
    
    Args:
        min_similarity: Minimum similarity threshold
        
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "retrieve_handbook_info",
            "description": "Retrieve information from the general customer handbook. Use this tool to answer questions about company policies, product offerings, refund policies, shipping information, and general company information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant handbook information"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }


def execute_handbook_retrieval(
    query: str, 
    k: int = 3, 
    min_similarity: float = 0.75,
    vectorstore=None
) -> tuple[str, List[Document]]:
    """
    Execute handbook retrieval.
    
    Args:
        query: Search query
        k: Number of results
        min_similarity: Minimum similarity threshold
        vectorstore: Optional pre-initialized vectorstore (for performance)
        
    Returns:
        Tuple of (serialized content, list of documents with metadata)
    """
    # Use provided vectorstore or initialize new one
    if vectorstore is None:
        handbook_store = EmbeddingStore(
            persist_directory="data/vector_store",
            collection_name="general_handbook",
            clear_existing=False
        )
        vectorstore = handbook_store.get_vectorstore()
    
    # Use similarity_search_with_score to get scores
    results_with_scores = vectorstore.similarity_search_with_score(query, k)
    
    # Filter by similarity threshold using utility function
    filtered_docs_with_scores = filter_by_similarity_threshold(results_with_scores, min_similarity, k)
    
    # Extract just the documents for serialization
    filtered_docs = [doc for doc, _ in filtered_docs_with_scores]
    
    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('handbook_name', 'Handbook')}\n"
            f"Section: {doc.metadata.get('Header 1', '')} {doc.metadata.get('Header 2', '')}\n"
            f"Content: {doc.page_content}"
        )
        for doc in filtered_docs
    )
    
    # Return (serialized content, list of (doc, similarity) tuples)
    return (serialized if serialized else "No relevant information found.", filtered_docs_with_scores)


def get_product_search_function(min_similarity: float = 0.75) -> Dict[str, Any]:
    """
    Get OpenAI function definition for product search.
    
    Args:
        min_similarity: Minimum similarity threshold
        
    Returns:
        OpenAI function definition
    """
    # Valid categories and brands from products.json
    VALID_CATEGORIES = ["Accessories", "Clothing", "Electronics"]
    VALID_BRANDS = [
        "ASUS", "Adidas", "Allbirds", "Anker", "Apple", "Bose", "Canon",
        "Carhartt", "Champion", "DJI", "Dell", "Fossil", "Garmin", "Google",
        "HP", "Herschel", "JBL", "Keychron", "Levi's", "Logitech", "Lululemon",
        "Nike", "Nintendo", "Oakley", "OnePlus", "Patagonia", "Ray-Ban", "Razer",
        "Samsung", "Sony", "SteelSeries", "The North Face", "Tommy Hilfiger",
        "Uniqlo", "Vans"
    ]
    
    return {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products using semantic search. Extract ALL relevant filters from user queries (price ranges, categories, brands, featured status) and apply them. Use this tool ONLY when the user wants to find/browse products and not when adding to cart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query describing what the user is looking for. Extract all filters (price, category, brand, featured) from the user's query."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    },
                    "category": {
                        "type": "string",
                        "enum": VALID_CATEGORIES,
                        "description": "Optional category filter. Must be one of: Accessories, Clothing, Electronics. Extract from user query if mentioned (e.g., 'laptops'/'phones'/'watches'/'smartwatch' → 'Electronics', 'shoes'/'clothes' → 'Clothing', 'headphones' → 'Accessories')."
                    },
                    "brand": {
                        "type": "string",
                        "enum": VALID_BRANDS,
                        "description": "Optional brand filter. Must be one of the valid brands. Extract from user query if a specific brand is mentioned (e.g., 'Apple', 'Nike', 'Samsung')."
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Optional maximum price filter. Extract from user queries like 'below $50', 'under $100', 'less than $200', 'cheap', 'affordable', 'budget'. Convert to numeric value (e.g., 'below $50' = 50)."
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Optional minimum price filter. Extract from user queries like 'above $50', 'over $100', 'more than $200', 'premium', 'high-end'. Convert to numeric value."
                    },
                    "is_featured": {
                        "type": "boolean",
                        "description": "Optional featured filter. Set to true if user asks for 'featured products', 'popular items', 'best sellers', 'trending products'."
                    }
                },
                "required": ["query"]
            }
        }
    }


def execute_product_search(
    query: str,
    k: int = 3,
    category: str = None,
    brand: str = None,
    min_price: float = None,
    max_price: float = None,
    is_featured: bool = None,
    min_similarity: float = 0.75,
    vectorstore=None
) -> tuple[str, List[Document]]:
    """
    Execute product search with optional filters.

    Args:
        query: Search query
        k: Number of results
        category: Optional category filter (must be one of: Accessories, Clothing, Electronics)
        brand: Optional brand filter (must be a valid brand from products.json)
        min_price: Optional minimum price filter
        max_price: Optional maximum price filter
        is_featured: Optional featured filter (True for featured products)
        min_similarity: Minimum similarity threshold
        vectorstore: Optional pre-initialized vectorstore (for performance)

    Returns:
        Tuple of (serialized content, list of documents with metadata)
    """
    # Use provided vectorstore or initialize new one
    if vectorstore is None:
        products_store = EmbeddingStore(
            persist_directory="data/vector_store",
            collection_name="products",
            clear_existing=False
        )
        vectorstore = products_store.get_vectorstore()
    
    # Build filter for exact matches (category, brand, is_featured) - ChromaDB supports these
    # ChromaDB requires $and operator when multiple conditions are present
    filter_conditions = []
    if category:
        filter_conditions.append({"category": category})
    if brand:
        filter_conditions.append({"brand": brand})
    if is_featured is not None:
        filter_conditions.append({"is_featured": is_featured})
    
    # Construct ChromaDB filter format
    # Single condition: {"category": "laptops"}
    # Multiple conditions: {"$and": [{"category": "laptops"}, {"brand": "Apple"}]}
    chroma_filter = None
    if len(filter_conditions) == 1:
        chroma_filter = filter_conditions[0]
    elif len(filter_conditions) > 1:
        chroma_filter = {"$and": filter_conditions}
    
    # Use similarity_search_with_score to get scores
    # Fetch more results initially to account for post-filtering (price)
    has_post_filters = (min_price is not None or max_price is not None)
    fetch_k = k * 3 if has_post_filters else k
    
    if chroma_filter:
        results_with_scores = vectorstore.similarity_search_with_score(
            query,
            k=fetch_k,
            filter=chroma_filter
        )
    else:
        results_with_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)
    
    # Filter by similarity threshold using utility function
    filtered_docs_with_scores = filter_by_similarity_threshold(results_with_scores, min_similarity, fetch_k)
    
    # Apply post-filters (price) - ChromaDB doesn't support range queries, so filter in Python
    if min_price is not None or max_price is not None:
        post_filtered = []
        for doc, similarity in filtered_docs_with_scores:
            # Price filtering
            price = doc.metadata.get("price")
            if price is not None:
                try:
                    price_float = float(price)
                    # Check price range
                    if min_price is not None and price_float < min_price:
                        continue
                    if max_price is not None and price_float > max_price:
                        continue
                except (ValueError, TypeError):
                    # If price can't be converted, skip price filtering for this item
                    pass
            elif min_price is not None:
                # If min_price is set but product has no price, exclude it
                continue
            
            post_filtered.append((doc, similarity))
        
        # Limit to k results after post-filtering
        filtered_docs_with_scores = post_filtered[:k]
    
    # Extract just the documents for serialization
    filtered_docs = [doc for doc, _ in filtered_docs_with_scores]
    
    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Product ID: {doc.metadata.get('product_id', 'N/A')}\n"
            f"Brand: {doc.metadata.get('brand', 'N/A')}\n"
            f"Category: {doc.metadata.get('category', 'N/A')}\n"
            f"Price: ${doc.metadata.get('price', 'N/A')}\n"
            f"Content: {doc.page_content}"
        )
        for doc in filtered_docs
    )
    
    # Return (serialized content, list of (doc, similarity) tuples)
    return (serialized if serialized else "No products found matching your criteria.", filtered_docs_with_scores)
