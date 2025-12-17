"""Order-related database models."""
from sqlalchemy import Column, Integer, String, Numeric, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from data.database.connection import Base


class Voucher(Base):
    """Voucher model for discount/payment vouchers."""
    
    __tablename__ = "vouchers"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    amount = Column(Numeric(10, 2), nullable=False)  # Voucher value in USD
    is_used = Column(Boolean, default=False, nullable=False, index=True)
    used_at = Column(DateTime(timezone=True), nullable=True)
    generated_by_session = Column(String(100), nullable=True, index=True)  # Session ID that generated it
    used_by_session = Column(String(100), nullable=True)  # Session ID that used it
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Optional expiration
    
    def __repr__(self):
        return f"<Voucher(id={self.id}, code='{self.code}', amount={self.amount}, is_used={self.is_used})>"


class Order(Base):
    """Order model representing customer purchases."""
    
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)  # User session identifier
    voucher_code = Column(String(50), ForeignKey("vouchers.code"), nullable=True)
    total_amount = Column(Numeric(10, 2), nullable=False)
    status = Column(String(50), default="completed", nullable=False)  # completed, failed, pending
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    voucher = relationship("Voucher", backref="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Order(id={self.id}, session_id='{self.session_id}', total={self.total_amount}, status='{self.status}')>"


class OrderItem(Base):
    """Order item model representing individual products in an order."""
    
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    product_name = Column(String(255), nullable=False)  # Snapshot of product name at time of order
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Numeric(10, 2), nullable=False)  # Snapshot of price at time of order
    subtotal = Column(Numeric(10, 2), nullable=False)  # quantity * unit_price
    
    # Relationships
    order = relationship("Order", back_populates="items")
    product = relationship("Product")
    
    def __repr__(self):
        return f"<OrderItem(id={self.id}, product_id={self.product_id}, quantity={self.quantity}, subtotal={self.subtotal})>"


class ShippingInfo(Base):
    """Shipping information model for user addresses."""
    
    __tablename__ = "shipping_info"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)  # User session identifier
    full_name = Column(String(255), nullable=False)
    address = Column(String(500), nullable=False)
    city = Column(String(100), nullable=False)
    zip_code = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<ShippingInfo(id={self.id}, session_id='{self.session_id}', full_name='{self.full_name}')>"

