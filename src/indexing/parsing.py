"""Product parsing module for loading and parsing products from database."""
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from data.database.connection import SessionLocal
from data.database.product_model import Product
from langchain_core.documents import Document


class ProductParser:
    """Intelligently loads and parses products from the database."""
    
    def __init__(self, db: Optional[Session] = None):
        """Initialize the parser with an optional database session."""
        self.db = db
        self._should_close_db = db is None
    
    def __enter__(self):
        """Context manager entry."""
        if self.db is None:
            self.db = SessionLocal()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_db and self.db:
            self.db.close()
    
    def load_products(
        self, 
        limit: Optional[int] = None,
        offset: int = 0,
        is_active_only: bool = True
    ) -> List[Product]:
        """
        Load products from the database.
        
        Args:
            limit: Maximum number of products to load (None for all)
            offset: Number of products to skip
            is_active_only: Only load active products
            
        Returns:
            List of Product objects
        """
        query = select(Product)
        
        if is_active_only:
            query = query.where(Product.is_active == True)
        
        query = query.offset(offset)
        
        if limit:
            query = query.limit(limit)
        
        result = self.db.execute(query)
        products = result.scalars().all()
        return list(products)
    
    def parse_product(self, product: Product) -> Dict[str, Any]:
        """
        Parse a single product into a structured dictionary.
        
        Only indexes properties useful for semantic search and filtering.
        Frequently changing properties (like stock_quantity) are excluded
        and should be fetched from the database after semantic search.
        
        Args:
            product: Product model instance
            
        Returns:
            Dictionary with parsed product data
        """
        # Build product text for semantic search
        # Only include properties that are useful for search and don't change frequently
        text_parts = []
        
        # Core product information (searchable)
        text_parts.append(f"Product Name: {product.name}")
        text_parts.append(f"Brand: {product.brand or 'Unknown'}")
        text_parts.append(f"Category: {product.category or 'Uncategorized'}")
        
        # Detailed description (main content for semantic search)
        text_parts.append(f"Description: {product.description}")
        
        # Tags (important for search)
        if product.tags:
            tags_text = ", ".join(product.tags)
            text_parts.append(f"Tags: {tags_text}")
        
        # Status indicators (for context, not frequently changing)
        if product.is_featured:
            text_parts.append("Featured Product")
        
        # Combine all text
        full_text = "\n".join(text_parts)
        
        # Build metadata for filtering and retrieval
        # Only include properties useful for filtering that don't change frequently
        metadata = {
            "product_id": product.id,  # Essential for fetching full product from DB
            "brand": product.brand or "",
            "category": product.category or "",
            "price": float(product.price),  # For price range filtering
            "is_active": product.is_active,  # For filtering active products
            "is_featured": product.is_featured,  # For filtering featured products
            "primary_image": product.primary_image or "",  # Primary image URL
        }
        
        # Add tags to metadata for filtering (convert list to comma-separated string)
        if product.tags:
            metadata["tags"] = ", ".join(product.tags)
        
        # NOTE: Excluded from index:
        # - stock_quantity: Changes frequently with purchases
        # - cost_price: Internal business data, not for search
        # - low_stock_threshold: Internal inventory management
        # - weight, dimensions: Not useful for semantic search
        # - SKU: Can be searched directly in database
        # - timestamps: Not useful for search
        
        return {
            "text": full_text,
            "metadata": metadata,
            "product_id": product.id
        }
    
    def parse_all_products(
        self,
        batch_size: int = 100,
        is_active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Parse all products from the database in batches.
        
        Args:
            batch_size: Number of products to process per batch
            is_active_only: Only parse active products
            
        Returns:
            List of parsed product dictionaries
        """
        parsed_products = []
        offset = 0
        
        while True:
            products = self.load_products(
                limit=batch_size,
                offset=offset,
                is_active_only=is_active_only
            )
            
            if not products:
                break
            
            for product in products:
                parsed = self.parse_product(product)
                parsed_products.append(parsed)
            
            offset += batch_size
            
            # If we got fewer products than batch_size, we're done
            if len(products) < batch_size:
                break
        
        return parsed_products
    
    def to_langchain_documents(
        self,
        parsed_products: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Convert parsed products to LangChain Document objects.
        
        Args:
            parsed_products: List of parsed product dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for parsed in parsed_products:
            doc = Document(
                page_content=parsed["text"],
                metadata={
                    **parsed["metadata"],
                    "source": "product",
                    "source_id": parsed["product_id"]
                }
            )
            documents.append(doc)
        
        return documents


class HandbookParser:
    """Parser for markdown handbook files."""
    
    def __init__(self, handbook_path: str = "data/handbooks/general_handbook.md"):
        """
        Initialize the handbook parser.
        
        Args:
            handbook_path: Path to the markdown handbook file
        """
        self.handbook_path = Path(handbook_path)
    
    def load_handbook(self) -> str:
        """
        Load markdown content from handbook file.
        
        Returns:
            Markdown content as string
        """
        if not self.handbook_path.exists():
            raise FileNotFoundError(f"Handbook file not found: {self.handbook_path}")
        
        with open(self.handbook_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse_handbook(self, markdown_content: str) -> Dict[str, Any]:
        """
        Parse handbook markdown into structured format.
        
        Args:
            markdown_content: Raw markdown content
            
        Returns:
            Dictionary with parsed handbook data
        """
        return {
            "text": markdown_content,
            "metadata": {
                "source": "handbook",
                "handbook_name": self.handbook_path.stem,
                "handbook_path": str(self.handbook_path)
            }
        }
    
    def to_langchain_document(self, parsed_handbook: Dict[str, Any]) -> Document:
        """
        Convert parsed handbook to LangChain Document.
        
        Args:
            parsed_handbook: Parsed handbook dictionary
            
        Returns:
            LangChain Document object
        """
        return Document(
            page_content=parsed_handbook["text"],
            metadata=parsed_handbook["metadata"]
        )

