"""Shipping information schemas for API validation."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ShippingInfoBase(BaseModel):
    """Base shipping info schema with common fields."""
    full_name: str = Field(..., min_length=1, max_length=255, description="Full name for shipping")
    address: str = Field(..., min_length=1, max_length=500, description="Complete street address")
    city: str = Field(..., min_length=1, max_length=100, description="City name")
    zip_code: str = Field(..., min_length=1, max_length=20, description="Zip/postal code")
    
    @field_validator('full_name', 'address', 'city', 'zip_code')
    @classmethod
    def validate_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class ShippingInfoCreate(ShippingInfoBase):
    """Schema for creating shipping information."""
    pass


class ShippingInfoResponse(ShippingInfoBase):
    """Schema for shipping information response."""
    id: int
    session_id: str
    
    class Config:
        from_attributes = True

