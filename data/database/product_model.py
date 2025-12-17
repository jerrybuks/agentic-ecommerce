"""Product model for ecommerce platform."""
from sqlalchemy import Column, Integer, String, Numeric, Text, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from data.database.connection import Base


class Product(Base):
    """Product model representing items in the ecommerce store."""
    
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    sku = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)  # Detailed description for semantic search
    
    # Pricing
    price = Column(Numeric(10, 2), nullable=False)
    cost_price = Column(Numeric(10, 2), nullable=True)  # Cost to business
    
    # Inventory
    stock_quantity = Column(Integer, default=0, nullable=False)
    low_stock_threshold = Column(Integer, default=10, nullable=True)
    
    # Product details
    weight = Column(Numeric(8, 2), nullable=True)  # in grams or kg
    dimensions = Column(JSON, nullable=True)  # {"length": 10, "width": 5, "height": 3}
    category = Column(String(100), nullable=True, index=True)
    tags = Column(JSON, nullable=True)  # Array of tags ["electronics", "smartphone"]
    
    # Media
    images = Column(JSON, nullable=True)  # Array of image URLs
    primary_image = Column(String(500), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_featured = Column(Boolean, default=False, nullable=False, index=True)
    
    # Brand info
    brand = Column(String(100), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', sku='{self.sku}')>"

