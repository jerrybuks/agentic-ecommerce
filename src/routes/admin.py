"""Admin routes for product management."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from src.data.database.connection import get_db
from src.data.database.product_model import Product
from src.data.database.product_schema import ProductCreate, ProductUpdate, ProductResponse

router = APIRouter(prefix="/admin/products", tags=["admin"])


@router.post(
    "/",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new product",
    description="Create a new product in the ecommerce platform"
)
def create_product(
    product: ProductCreate,
    db: Session = Depends(get_db)
):
    """Create a new product."""
    # Check if SKU already exists
    existing_product = db.query(Product).filter(Product.sku == product.sku).first()
    if existing_product:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Product with SKU '{product.sku}' already exists"
        )
    
    # Create new product
    db_product = Product(**product.model_dump())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    
    return db_product


@router.get(
    "/",
    response_model=List[ProductResponse],
    summary="Get all products",
    description="Retrieve all products with optional filtering"
)
def get_products(
    skip: int = 0,
    limit: int = 100,
    is_active: Optional[bool] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all products with optional filters."""
    query = db.query(Product)
    
    # Apply filters
    if is_active is not None:
        query = query.filter(Product.is_active == is_active)
    if category:
        query = query.filter(Product.category == category)
    if brand:
        query = query.filter(Product.brand == brand)
    
    products = query.offset(skip).limit(limit).all()
    return products


@router.get(
    "/{product_id}",
    response_model=ProductResponse,
    summary="Get product by ID",
    description="Retrieve a specific product by its ID"
)
def get_product(
    product_id: int,
    db: Session = Depends(get_db)
):
    """Get a product by ID."""
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found"
        )
    return product


@router.get(
    "/sku/{sku}",
    response_model=ProductResponse,
    summary="Get product by SKU",
    description="Retrieve a specific product by its SKU"
)
def get_product_by_sku(
    sku: str,
    db: Session = Depends(get_db)
):
    """Get a product by SKU."""
    product = db.query(Product).filter(Product.sku == sku).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with SKU '{sku}' not found"
        )
    return product


@router.put(
    "/{product_id}",
    response_model=ProductResponse,
    summary="Update a product",
    description="Update an existing product by ID"
)
def update_product(
    product_id: int,
    product_update: ProductUpdate,
    db: Session = Depends(get_db)
):
    """Update a product."""
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found"
        )
    
    # Check if SKU is being updated and if it conflicts
    if product_update.sku and product_update.sku != db_product.sku:
        existing_product = db.query(Product).filter(Product.sku == product_update.sku).first()
        if existing_product:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Product with SKU '{product_update.sku}' already exists"
            )
    
    # Update only provided fields
    update_data = product_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_product, field, value)
    
    db.commit()
    db.refresh(db_product)
    
    return db_product


@router.patch(
    "/{product_id}",
    response_model=ProductResponse,
    summary="Partially update a product",
    description="Partially update an existing product by ID (alias for PUT)"
)
def patch_product(
    product_id: int,
    product_update: ProductUpdate,
    db: Session = Depends(get_db)
):
    """Partially update a product (same as PUT)."""
    return update_product(product_id, product_update, db)

