"""Cart state management for user sessions."""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CartItem:
    """Represents an item in the cart."""
    product_id: int
    product_name: str
    quantity: int
    unit_price: float
    primary_image: Optional[str] = None
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal for this cart item."""
        return float(self.unit_price * self.quantity)


class CartManager:
    """Manages shopping carts for user sessions."""
    
    def __init__(self):
        """Initialize cart manager with empty carts."""
        # session_id -> List[CartItem]
        self._carts: Dict[str, List[CartItem]] = defaultdict(list)
    
    def add_to_cart(
        self,
        session_id: str,
        product_id: int,
        product_name: str,
        quantity: int,
        unit_price: float,
        primary_image: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Add a product to the cart. Only adds if item doesn't already exist.
        
        Args:
            session_id: User session identifier
            product_id: Product ID to add
            product_name: Product name
            quantity: Quantity to add
            unit_price: Price per unit
            primary_image: Optional product image URL
            
        Returns:
            Dictionary with cart status and message
        """
        cart = self._carts[session_id]
        
        # Check if product already in cart
        for item in cart:
            if item.product_id == product_id:
                return {
                    "success": False,
                    "message": f"{product_name} is already in your cart. Use edit_item_in_cart to update the quantity.",
                    "cart_total": self.get_cart_total(session_id),
                    "item_count": len(cart)
                }
        
        # Add new item
        cart_item = CartItem(
            product_id=product_id,
            product_name=product_name,
            quantity=quantity,
            unit_price=unit_price,
            primary_image=primary_image
        )
        cart.append(cart_item)
        
        return {
            "success": True,
            "message": f"Added {quantity}x {product_name} to cart",
            "cart_total": self.get_cart_total(session_id),
            "item_count": len(cart)
        }
    
    def edit_item_in_cart(
        self,
        session_id: str,
        product_id: int,
        quantity: int
    ) -> Dict[str, any]:
        """
        Update the quantity of an item in the cart.
        
        Args:
            session_id: User session identifier
            product_id: Product ID to update
            quantity: New quantity (must be > 0)
            
        Returns:
            Dictionary with cart status and message
        """
        cart = self._carts.get(session_id, [])
        
        if quantity <= 0:
            return {
                "success": False,
                "message": f"Quantity must be greater than 0. Use remove_from_cart to remove items.",
                "cart_total": self.get_cart_total(session_id),
                "item_count": len(cart)
            }
        
        # Find item in cart
        for item in cart:
            if item.product_id == product_id:
                item.quantity = quantity
                return {
                    "success": True,
                    "message": f"Updated {item.product_name} quantity to {quantity}",
                    "cart_total": self.get_cart_total(session_id),
                    "item_count": len(cart)
                }
        
        return {
            "success": False,
            "message": f"Product with ID {product_id} not found in cart.",
            "cart_total": self.get_cart_total(session_id),
            "item_count": len(cart)
        }
    
    def remove_from_cart(
        self,
        session_id: str,
        product_id: int
    ) -> Dict[str, any]:
        """
        Remove an item from the cart.
        
        Args:
            session_id: User session identifier
            product_id: Product ID to remove
            
        Returns:
            Dictionary with cart status and message
        """
        cart = self._carts.get(session_id, [])
        
        # Find and remove item
        for i, item in enumerate(cart):
            if item.product_id == product_id:
                product_name = item.product_name
                cart.pop(i)
                return {
                    "success": True,
                    "message": f"Removed {product_name} from cart",
                    "cart_total": self.get_cart_total(session_id),
                    "item_count": len(cart)
                }
        
        return {
            "success": False,
            "message": f"Product with ID {product_id} not found in cart.",
            "cart_total": self.get_cart_total(session_id),
            "item_count": len(cart)
        }
    
    def get_cart(self, session_id: str) -> List[CartItem]:
        """
        Get all items in the cart for a session.
        
        Args:
            session_id: User session identifier
            
        Returns:
            List of cart items
        """
        return self._carts.get(session_id, [])
    
    def get_cart_total(self, session_id: str) -> float:
        """
        Calculate total amount for cart.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Total cart amount
        """
        cart = self._carts.get(session_id, [])
        return sum(item.subtotal for item in cart)
    
    def clear_cart(self, session_id: str):
        """
        Clear all items from cart.
        
        Args:
            session_id: User session identifier
        """
        if session_id in self._carts:
            self._carts[session_id] = []
    
    def get_cart_summary(self, session_id: str) -> Dict[str, any]:
        """
        Get formatted cart summary.
        
        Args:
            session_id: User session identifier
            
        Returns:
            Dictionary with cart summary
        """
        cart = self.get_cart(session_id)
        total = self.get_cart_total(session_id)
        
        items = [
            {
                "product_id": item.product_id,
                "product_name": item.product_name,
                "quantity": item.quantity,
                "unit_price": float(item.unit_price),
                "subtotal": float(item.subtotal),
                "primary_image": item.primary_image
            }
            for item in cart
        ]
        
        return {
            "items": items,
            "item_count": len(cart),
            "total": float(total),
            "total_formatted": f"${total:.2f}"
        }


# Global cart manager instance
cart_manager = CartManager()

