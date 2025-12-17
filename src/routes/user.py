"""User routes for querying and voucher management."""
import hashlib
import secrets
from fastapi import APIRouter, HTTPException, Request, Depends, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.querying.service import QueryService
from src.config import settings
from data.database.connection import get_db
from data.database.order_models import Voucher, Order, OrderItem
from data.database.product_model import Product
from data.database.product_schema import ProductResponse
from src.utils.cart import cart_manager

router = APIRouter(prefix="/user", tags=["user"])

# Query service will be initialized with vector stores from app state
query_service = None


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    Handles proxies and forwarded headers.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address as string
    """
    # Check for forwarded IP (when behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, take the first one
        ip = forwarded_for.split(",")[0].strip()
        return ip
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client IP
    if request.client:
        return request.client.host
    
    return "unknown"


def generate_session_id(ip_address: str) -> str:
    """
    Generate a consistent session_id from IP address.
    Uses hash to ensure consistent ID for same IP.
    
    Args:
        ip_address: Client IP address
        
    Returns:
        Session ID based on IP address
    """
    # Hash the IP address to create a consistent session_id
    # This ensures same IP always gets same session_id
    hash_obj = hashlib.md5(ip_address.encode())
    return f"session_{hash_obj.hexdigest()[:16]}"


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User's question or request")
    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        example=settings.default_similarity_threshold,
        description=f"Minimum similarity score threshold for retrieval (default: {settings.default_similarity_threshold} if not provided, range: 0.0-1.0)"
    )


class SourceResponse(BaseModel):
    """Source document response model."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    similarity: float = Field(..., description="Similarity score (0.0-1.0)")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    input: Dict[str, Any] = Field(..., description="Query parameters used in product search (if applicable)")
    answer: str = Field(..., description="Agent's answer/response content")
    agents_used: List[str] = Field(..., description="List of agents used in processing")
    routing_mode: str = Field(..., description="Routing mode: single, sequential, parallel, or direct")
    sources: List[SourceResponse] = Field(default_factory=list, description="Retrieved source documents")
    session_id: str = Field(..., description="Session ID for conversation context")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request):
    """
    Process a user query through the multi-agent system.
    
    The orchestrator agent will route the query to the appropriate sub-agent:
    - General Info Agent: For company policies, FAQs, general information
    - Order Agent: For product search, orders, purchasing
    
    Conversation context is automatically maintained based on the user's IP address.
    Users from the same IP will have their conversation history preserved.
    
    Args:
        request: Query request with user query
        http_request: FastAPI request object (for IP extraction)
        
    Returns:
        Query response with agent's answer
    """
    global query_service
    
    try:
        # Initialize query service with vector stores from app state (lazy initialization)
        if query_service is None:
            from src.querying.service import QueryService
            query_service = QueryService(
                handbook_vectorstore=http_request.app.state.handbook_vectorstore,
                products_vectorstore=http_request.app.state.products_vectorstore
            )
        
        # Extract IP address and generate session_id
        client_ip = get_client_ip(http_request)
        session_id = generate_session_id(client_ip)
        
        # Process query with auto-generated session_id and min_similarity
        result = await query_service.query(
            user_query=request.query,
            session_id=session_id,
            min_similarity=request.min_similarity
        )
        
        # Format sources (sources are always (doc, similarity) tuples)
        sources = []
        for doc, similarity in result.get("sources", []):
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata.copy(),
                "similarity": float(similarity)
            })
        
        # Get query_params, or use user's query if empty
        query_params = result.get("query_params", {})
        if not query_params:
            query_params = {"query": request.query}
        
        return QueryResponse(
            input=query_params,
            answer=result["response"],
            agents_used=result.get("agents_used", []),
            routing_mode=result.get("routing_mode", "single"),
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


# Voucher endpoints
class VoucherResponse(BaseModel):
    """Voucher response model."""
    id: int
    code: str
    amount: float
    is_used: bool
    
    class Config:
        from_attributes = True


@router.post(
    "/vouchers/generate",
    response_model=VoucherResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new voucher",
    description="Generate a new $2000 USD voucher code. Returns existing unused voucher if user already has one."
)
def generate_voucher(http_request: Request, db: Session = Depends(get_db)):
    """Generate a new $2000 USD voucher or return existing unused one."""
    # Get session_id for the user
    client_ip = get_client_ip(http_request)
    session_id = generate_session_id(client_ip)
    
    # Check if user already has an unused voucher
    existing_unused = db.query(Voucher).filter(
        Voucher.generated_by_session == session_id,
        Voucher.is_used == False
    ).first()
    
    # If user has an unused voucher, return it
    if existing_unused:
        return existing_unused
    
    # Generate a unique voucher code
    while True:
        code = f"VOUCHER-{secrets.token_hex(8).upper()}"
        existing = db.query(Voucher).filter(Voucher.code == code).first()
        if not existing:
            break
    
    # Create voucher with $2000 value
    voucher = Voucher(
        code=code,
        amount=2000.00,
        is_used=False,
        generated_by_session=session_id
    )
    db.add(voucher)
    db.commit()
    db.refresh(voucher)
    
    return voucher


# Cart endpoints
class CartItemResponse(BaseModel):
    """Cart item response model."""
    product_id: int
    product_name: str
    quantity: int
    unit_price: float
    subtotal: float
    primary_image: Optional[str] = None


class CartResponse(BaseModel):
    """Cart response model."""
    items: List[CartItemResponse]
    item_count: int
    total: float
    total_formatted: str


@router.get("/cart", response_model=CartResponse, summary="Get user's cart")
def get_cart(http_request: Request):
    """
    Get the current user's shopping cart.
    
    Returns all items in the cart, including quantities, prices, and totals.
    Cart is maintained in-memory based on the user's session ID (derived from IP address).
    
    Returns:
        Cart response with items and total
    """
    client_ip = get_client_ip(http_request)
    session_id = generate_session_id(client_ip)
    
    summary = cart_manager.get_cart_summary(session_id)
    
    return CartResponse(
        items=[CartItemResponse(**item) for item in summary["items"]],
        item_count=summary["item_count"],
        total=summary["total"],
        total_formatted=summary["total_formatted"]
    )


# Order endpoints
class OrderItemResponse(BaseModel):
    """Order item response model."""
    id: int
    product_id: int
    product_name: str
    quantity: int
    unit_price: float
    subtotal: float
    
    class Config:
        from_attributes = True


class OrderResponse(BaseModel):
    """Order response model."""
    id: int
    session_id: str
    voucher_code: Optional[str] = None
    total_amount: float
    status: str
    created_at: str
    items: List[OrderItemResponse]
    
    class Config:
        from_attributes = True


@router.get("/orders", response_model=List[OrderResponse], summary="Get user's orders")
def get_orders(http_request: Request, db: Session = Depends(get_db)):
    """
    Get all orders for the current user.
    
    Returns a list of all orders placed by the user, including order items,
    voucher codes used, and order status. Orders are sorted by creation date (newest first).
    
    Returns:
        List of order responses
    """
    client_ip = get_client_ip(http_request)
    session_id = generate_session_id(client_ip)
    
    # Query orders for this session, ordered by creation date (newest first)
    orders = db.query(Order).filter(
        Order.session_id == session_id
    ).order_by(Order.created_at.desc()).all()
    
    # Build response with order items
    order_responses = []
    for order in orders:
        order_items = [
            OrderItemResponse(
                id=item.id,
                product_id=item.product_id,
                product_name=item.product_name,
                quantity=item.quantity,
                unit_price=float(item.unit_price),
                subtotal=float(item.subtotal)
            )
            for item in order.items
        ]
        
        order_responses.append(OrderResponse(
            id=order.id,
            session_id=order.session_id,
            voucher_code=order.voucher_code,
            total_amount=float(order.total_amount),
            status=order.status,
            created_at=order.created_at.isoformat(),
            items=order_items
        ))
    
    return order_responses


# Product endpoints
class ProductListResponse(BaseModel):
    """Product list response model."""
    products: List[ProductResponse]
    total: int
    page: int
    page_size: int


@router.get("/products", response_model=ProductListResponse, summary="Get products with search and filters")
def get_products(
    http_request: Request,
    db: Session = Depends(get_db),
    search: Optional[str] = Query(None, description="Search query for product name, description, or SKU"),
    category: Optional[str] = Query(None, description="Filter by category"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags to filter by"),
    is_featured: Optional[bool] = Query(None, description="Filter by featured status"),
    is_active: Optional[bool] = Query(True, description="Filter by active status (default: True)"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page")
):
    """
    Get products with search and filtering capabilities.
    
    Supports:
    - Text search across product name, description, and SKU
    - Filtering by category, brand, price range, tags, featured status, and active status
    - Pagination with configurable page size
    
    Args:
        search: Search query (searches in name, description, SKU)
        category: Filter by category
        brand: Filter by brand
        min_price: Minimum price
        max_price: Maximum price
        tags: Comma-separated tags (product must have at least one)
        is_featured: Filter by featured status
        is_active: Filter by active status (default: True)
        page: Page number (default: 1)
        page_size: Items per page (default: 20, max: 100)
    
    Returns:
        Paginated list of products matching the criteria
    """
    # Start with base query
    query = db.query(Product)
    
    # Apply active filter (default to True if not specified)
    if is_active is not None:
        query = query.filter(Product.is_active == is_active)
    
    # Apply text search (name, description, or SKU)
    if search:
        search_term = f"%{search.lower()}%"
        query = query.filter(
            or_(
                Product.name.ilike(search_term),
                Product.description.ilike(search_term),
                Product.sku.ilike(search_term)
            )
        )
    
    # Apply category filter
    if category:
        query = query.filter(Product.category == category)
    
    # Apply brand filter
    if brand:
        query = query.filter(Product.brand == brand)
    
    # Apply price range filters
    if min_price is not None:
        query = query.filter(Product.price >= min_price)
    if max_price is not None:
        query = query.filter(Product.price <= max_price)
    
    # Apply featured filter
    if is_featured is not None:
        query = query.filter(Product.is_featured == is_featured)
    
    # Apply tags filter (product must have at least one of the specified tags)
    # Note: Tags filtering is done in Python after applying other filters
    # for compatibility with PostgreSQL JSON columns. This could be optimized
    # later with JSONB queries for better performance.
    apply_tags_filter = False
    tag_list = []
    if tags:
        tag_list = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
        apply_tags_filter = len(tag_list) > 0
    
    # If tags filter is needed, fetch all matching products (after other filters)
    # then filter by tags in Python, then paginate
    if apply_tags_filter:
        # Apply all other filters first, then filter tags in Python
        all_matching_products = query.order_by(Product.created_at.desc()).all()
        # Filter by tags (case-insensitive match)
        filtered_products = [
            p for p in all_matching_products
            if p.tags and isinstance(p.tags, list) and
            any(requested_tag in [t.lower() for t in p.tags] for requested_tag in tag_list)
        ]
        total = len(filtered_products)
        
        # Apply pagination to filtered results
        offset = (page - 1) * page_size
        products = filtered_products[offset:offset + page_size]
    else:
        # No tags filter, use normal pagination
        total = query.count()
        offset = (page - 1) * page_size
        products = query.order_by(Product.created_at.desc()).offset(offset).limit(page_size).all()
    
    return ProductListResponse(
        products=[ProductResponse.model_validate(product) for product in products],
        total=total,
        page=page,
        page_size=page_size
    )


