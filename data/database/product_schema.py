"""Product schemas for API validation."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime


class ProductBase(BaseModel):
    """Base product schema with common fields."""
    name: str = Field(..., min_length=1, max_length=255, description="Product name")
    sku: str = Field(..., min_length=1, max_length=100, description="Stock Keeping Unit")
    description: str = Field(..., min_length=10, description="Detailed product description for semantic search")
    price: Decimal = Field(..., gt=0, description="Product price")
    cost_price: Optional[Decimal] = Field(None, ge=0, description="Cost to business")
    stock_quantity: int = Field(0, ge=0, description="Available stock quantity")
    low_stock_threshold: Optional[int] = Field(10, ge=0, description="Alert threshold for low stock")
    weight: Optional[Decimal] = Field(None, ge=0, description="Product weight")
    dimensions: Optional[Dict[str, Any]] = Field(None, description="Product dimensions")
    category: Optional[str] = Field(None, max_length=100, description="Product category")
    tags: Optional[List[str]] = Field(None, description="Product tags")
    images: Optional[List[str]] = Field(None, description="Array of image URLs")
    primary_image: Optional[str] = Field(None, max_length=500, description="Primary product image URL")
    is_active: bool = Field(True, description="Whether product is active/available")
    is_featured: bool = Field(False, description="Whether product is featured")
    brand: Optional[str] = Field(None, max_length=100, description="Product brand")
    
    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v):
        if v is not None:
            allowed_keys = {'length', 'width', 'height', 'unit'}
            if not all(key in allowed_keys for key in v.keys()):
                raise ValueError(f"Dimensions can only contain: {allowed_keys}")
        return v


class ProductCreate(ProductBase):
    """Schema for creating a new product."""
    pass


class ProductUpdate(BaseModel):
    """Schema for updating a product (all fields optional)."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    sku: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, min_length=10)
    price: Optional[Decimal] = Field(None, gt=0)
    cost_price: Optional[Decimal] = Field(None, ge=0)
    stock_quantity: Optional[int] = Field(None, ge=0)
    low_stock_threshold: Optional[int] = Field(None, ge=0)
    weight: Optional[Decimal] = Field(None, ge=0)
    dimensions: Optional[Dict[str, Any]] = None
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None
    images: Optional[List[str]] = None
    primary_image: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    is_featured: Optional[bool] = None
    brand: Optional[str] = Field(None, max_length=100)
    
    @field_validator('dimensions')
    @classmethod
    def validate_dimensions(cls, v):
        if v is not None:
            allowed_keys = {'length', 'width', 'height', 'unit'}
            if not all(key in allowed_keys for key in v.keys()):
                raise ValueError(f"Dimensions can only contain: {allowed_keys}")
        return v


class ProductResponse(ProductBase):
    """Schema for product response."""
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

