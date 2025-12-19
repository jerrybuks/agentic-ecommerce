"""Order management tools using OpenAI function calling."""
import json
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import joinedload
from data.database.connection import SessionLocal
from data.database.product_model import Product
from data.database.order_models import Order, OrderItem, Voucher, ShippingInfo
from data.database.shipping_schema import ShippingInfoCreate
from src.utils.cart import cart_manager


def get_add_to_cart_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for adding items to cart.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add a new product to cart. Only for products NOT already in cart. If product exists, use edit_item_in_cart. Check cart with view_cart first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The ID of the product to add to cart"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Quantity of the product to add",
                        "default": 1
                    }
                },
                "required": ["product_id"]
            }
        }
    }


def execute_add_to_cart(session_id: str, product_id: int, quantity: int = 1) -> str:
    """
    Execute adding product to cart.
    
    Args:
        session_id: User session identifier
        product_id: Product ID to add
        quantity: Quantity to add
        
    Returns:
        Result message
    """
    db = SessionLocal()
    try:
        # Fetch product details
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            return f"Error: Product with ID {product_id} not found."
        
        if not product.is_active:
            return f"Error: Product '{product.name}' is not available for purchase."
        
        if product.stock_quantity < quantity:
            return f"Error: Insufficient stock. Only {product.stock_quantity} available for '{product.name}'."
        
        # Add to cart
        result = cart_manager.add_to_cart(
            session_id=session_id,
            product_id=product_id,
            product_name=product.name,
            quantity=quantity,
            unit_price=float(product.price),
            primary_image=product.primary_image
        )
        
        if result["success"]:
            return result["message"] + f" Cart total: ${result['cart_total']:.2f}"
        else:
            return result["message"]
    finally:
        db.close()


def get_edit_item_in_cart_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for editing item quantity in cart.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "edit_item_in_cart",
            "description": "Update quantity of item already in cart. Use when: changing quantity, adding to existing (new_quantity = current + X), or reducing (new_quantity = current - X). Check cart with view_cart first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The ID of the product in the cart to update"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "The new quantity for this item (must be greater than 0)"
                    }
                },
                "required": ["product_id", "quantity"]
            }
        }
    }


def execute_edit_item_in_cart(session_id: str, product_id: int, quantity: int) -> str:
    """
    Execute editing item quantity in cart.
    
    Args:
        session_id: User session identifier
        product_id: Product ID to update
        quantity: New quantity
        
    Returns:
        Result message
    """
    result = cart_manager.edit_item_in_cart(
        session_id=session_id,
        product_id=product_id,
        quantity=quantity
    )
    
    if result["success"]:
        return result["message"] + f" Cart total: ${result['cart_total']:.2f}"
    else:
        return result["message"]


def get_remove_from_cart_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for removing items from cart.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": "Completely remove item from cart. Only for full removal. If user says 'remove X items', use edit_item_in_cart to reduce quantity instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The ID of the product to remove from cart"
                    }
                },
                "required": ["product_id"]
            }
        }
    }


def execute_remove_from_cart(session_id: str, product_id: int) -> str:
    """
    Execute removing item from cart.
    
    Args:
        session_id: User session identifier
        product_id: Product ID to remove
        
    Returns:
        Result message
    """
    result = cart_manager.remove_from_cart(
        session_id=session_id,
        product_id=product_id
    )
    
    if result["success"]:
        return result["message"] + f" Cart total: ${result['cart_total']:.2f}"
    else:
        return result["message"]


def get_view_cart_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for viewing cart.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": "View cart contents with quantities, prices, and total. Always call this to get current cart state - do not describe from memory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }


def execute_view_cart(session_id: str) -> str:
    """
    Execute viewing cart contents.
    
    Args:
        session_id: User session identifier
        
    Returns:
        Formatted cart summary
    """
    summary = cart_manager.get_cart_summary(session_id)
    
    if summary["item_count"] == 0:
        return "Your cart is empty. Add items to your cart to get started!"
    
    # Format cart display
    lines = ["Your Shopping Cart:", ""]
    for item in summary["items"]:
        lines.append(
            f"• {item['product_name']} (ID: {item['product_id']})\n"
            f"  Quantity: {item['quantity']} × ${item['unit_price']:.2f} = ${item['subtotal']:.2f}"
        )
    
    lines.append("")
    lines.append(f"Total: {summary['total_formatted']}")
    lines.append(f"Items in cart: {summary['item_count']}")
    
    return "\n".join(lines)


def get_shipping_info_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for retrieving shipping information.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "get_shipping_info",
            "description": "Check if shipping information exists. Call when user asks about shipping info or before purchase.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }


def execute_get_shipping_info(session_id: str) -> str:
    """
    Execute getting shipping information for a session.
    
    Args:
        session_id: User session identifier
        
    Returns:
        Shipping info result message
    """
    db = SessionLocal()
    try:
        shipping_info = db.query(ShippingInfo).filter(
            ShippingInfo.session_id == session_id
        ).first()
        
        if shipping_info:
            return (
                f"Shipping information found:\n"
                f"Full Name: {shipping_info.full_name}\n"
                f"Address: {shipping_info.address}\n"
                f"City: {shipping_info.city}\n"
                f"Zip Code: {shipping_info.zip_code}\n"
                f"\nYou can proceed with purchase using a voucher code."
            )
        else:
            return (
                "No shipping information found. "
                "Please provide your shipping details (full name, address, city, zip code) before completing your purchase."
            )
    except Exception as e:
        return f"Error retrieving shipping information: {str(e)}"
    finally:
        db.close()


def get_create_shipping_info_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for creating shipping information.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "create_shipping_info",
            "description": "Create shipping information. Extract fullName, address, city, zipCode from user message. All fields required. Call this tool to save, don't just acknowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipping_data": {
                        "type": "object",
                        "description": "Extracted shipping information as JSON object with fields: fullName, address, city, zipCode",
                        "properties": {
                            "fullName": {
                                "type": "string",
                                "description": "Full name for shipping"
                            },
                            "address": {
                                "type": "string",
                                "description": "Complete street address"
                            },
                            "city": {
                                "type": "string",
                                "description": "City name"
                            },
                            "zipCode": {
                                "type": "string",
                                "description": "Zip/postal code"
                            }
                        },
                        "required": ["fullName", "address", "city", "zipCode"]
                    }
                },
                "required": ["shipping_data"]
            }
        }
    }


def execute_create_shipping_info(session_id: str, shipping_data: Dict[str, str]) -> str:
    """
    Execute creating shipping information.
    
    Args:
        session_id: User session identifier
        shipping_data: Dictionary with fullName, address, city, zipCode
        
    Returns:
        Creation result message
    """
    db = SessionLocal()
    try:
        # Validate using Pydantic schema
        try:
            shipping_create = ShippingInfoCreate(
                full_name=shipping_data.get("fullName", ""),
                address=shipping_data.get("address", ""),
                city=shipping_data.get("city", ""),
                zip_code=shipping_data.get("zipCode", "")
            )
        except Exception as e:
            return f"Error: Invalid shipping information. {str(e)}. Please provide all required fields: fullName, address, city, and zipCode."
        
        # Check if shipping info already exists
        existing = db.query(ShippingInfo).filter(
            ShippingInfo.session_id == session_id
        ).first()
        
        if existing:
            # Update existing
            existing.full_name = shipping_create.full_name
            existing.address = shipping_create.address
            existing.city = shipping_create.city
            existing.zip_code = shipping_create.zip_code
            message = "Shipping information updated successfully!"
        else:
            # Create new
            shipping_info = ShippingInfo(
                session_id=session_id,
                full_name=shipping_create.full_name,
                address=shipping_create.address,
                city=shipping_create.city,
                zip_code=shipping_create.zip_code
            )
            db.add(shipping_info)
            message = "Shipping information saved successfully!"
        
        db.commit()
        
        return (
            f"{message}\n"
            f"Full Name: {shipping_create.full_name}\n"
            f"Address: {shipping_create.address}\n"
            f"City: {shipping_create.city}\n"
            f"Zip Code: {shipping_create.zip_code}\n"
            f"\nYou can now proceed with purchase using a voucher code."
        )
    except Exception as e:
        db.rollback()
        return f"Error saving shipping information: {str(e)}"
    finally:
        db.close()


def get_edit_shipping_info_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for editing shipping information.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "edit_shipping_info",
            "description": "Update existing shipping information. Extract only fields to change (fullName, address, city, zipCode). Partial updates allowed. Call this tool to save changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shipping_data": {
                        "type": "object",
                        "description": "Updated shipping information as JSON object. Include only the fields the user wants to update: fullName, address, city, and/or zipCode. Partial updates are allowed.",
                        "properties": {
                            "fullName": {
                                "type": "string",
                                "description": "Full name for shipping (only include if user wants to update this)"
                            },
                            "address": {
                                "type": "string",
                                "description": "Complete street address (only include if user wants to update this)"
                            },
                            "city": {
                                "type": "string",
                                "description": "City name (only include if user wants to update this)"
                            },
                            "zipCode": {
                                "type": "string",
                                "description": "Zip/postal code (only include if user wants to update this)"
                            }
                        },
                        "required": []
                    }
                },
                "required": ["shipping_data"]
            }
        }
    }


def execute_edit_shipping_info(session_id: str, shipping_data: Dict[str, str]) -> str:
    """
    Execute editing shipping information with partial updates.
    
    Args:
        session_id: User session identifier
        shipping_data: Dictionary with any of: fullName, address, city, zipCode (partial updates allowed)
        
    Returns:
        Update result message
    """
    db = SessionLocal()
    try:
        # Check if shipping info exists
        existing = db.query(ShippingInfo).filter(
            ShippingInfo.session_id == session_id
        ).first()
        
        if not existing:
            return (
                "Error: No shipping information found to update. "
                "Please use create_shipping_info first to create your shipping information."
            )
        
        # Track which fields are being updated
        updated_fields = []
        
        # Update only provided fields with validation
        if "fullName" in shipping_data and shipping_data["fullName"]:
            full_name = shipping_data["fullName"].strip()
            if not full_name:
                return "Error: Full name cannot be empty."
            if len(full_name) > 255:
                return "Error: Full name must be 255 characters or less."
            existing.full_name = full_name
            updated_fields.append("Full Name")
        
        if "address" in shipping_data and shipping_data["address"]:
            address = shipping_data["address"].strip()
            if not address:
                return "Error: Address cannot be empty."
            if len(address) > 500:
                return "Error: Address must be 500 characters or less."
            existing.address = address
            updated_fields.append("Address")
        
        if "city" in shipping_data and shipping_data["city"]:
            city = shipping_data["city"].strip()
            if not city:
                return "Error: City cannot be empty."
            if len(city) > 100:
                return "Error: City must be 100 characters or less."
            existing.city = city
            updated_fields.append("City")
        
        if "zipCode" in shipping_data and shipping_data["zipCode"]:
            zip_code = shipping_data["zipCode"].strip()
            if not zip_code:
                return "Error: Zip code cannot be empty."
            if len(zip_code) > 20:
                return "Error: Zip code must be 20 characters or less."
            existing.zip_code = zip_code
            updated_fields.append("Zip Code")
        
        if not updated_fields:
            return "Error: No valid fields provided to update. Please specify at least one field: fullName, address, city, or zipCode."
        
        db.commit()
        db.refresh(existing)  # Refresh to ensure we have the latest data
        
        return (
            f"Shipping information updated successfully! Updated fields: {', '.join(updated_fields)}\n"
            f"Full Name: {existing.full_name}\n"
            f"Address: {existing.address}\n"
            f"City: {existing.city}\n"
            f"Zip Code: {existing.zip_code}\n"
            f"\nYou can now proceed with purchase using a voucher code."
        )
    except Exception as e:
        db.rollback()
        return f"Error updating shipping information: {str(e)}"
    finally:
        db.close()


def get_get_orders_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for retrieving orders.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": "Get order information. If order_id provided, returns that order. Otherwise returns 5 most recent orders. Always call this to get current state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "Optional order ID. If provided, returns that specific order. If not provided, returns the 5 most recent orders."
                    }
                },
                "required": []
            }
        }
    }


def execute_get_orders(session_id: str, order_id: Optional[int] = None) -> str:
    """
    Execute getting orders for a session.
    
    Args:
        session_id: User session identifier
        order_id: Optional specific order ID to retrieve
        
    Returns:
        Formatted orders information
    """
    db = SessionLocal()
    try:
        if order_id:
            # Get specific order with eager loading of items
            order = db.query(Order).options(
                joinedload(Order.items)
            ).filter(
                Order.id == order_id,
                Order.session_id == session_id
            ).first()
            
            if not order:
                return f"Error: Order ID {order_id} not found or does not belong to your session."
            
            # Format order with items
            lines = [
                f"Order #{order.id}",
                f"Status: {order.status}",
                f"Total: ${float(order.total_amount):.2f}",
                f"Voucher Code: {order.voucher_code if order.voucher_code else 'None'}",
                f"Created: {order.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Items:"
            ]
            
            for item in order.items:
                lines.append(
                    f"  • {item.product_name} (Product ID: {item.product_id})\n"
                    f"    Quantity: {item.quantity} × ${float(item.unit_price):.2f} = ${float(item.subtotal):.2f}"
                )
            
            return "\n".join(lines)
        else:
            # Get 5 most recent orders with eager loading of items to avoid N+1 queries
            orders = db.query(Order).options(
                joinedload(Order.items)
            ).filter(
                Order.session_id == session_id
            ).order_by(Order.created_at.desc()).limit(5).all()
            
            if not orders:
                return "You have no orders yet. Start shopping to create your first order!"
            
            lines = [f"Your {len(orders)} Most Recent Orders:", ""]
            
            for order in orders:
                lines.append(
                    f"Order #{order.id} - {order.status.upper()}\n"
                    f"Total: ${float(order.total_amount):.2f}\n"
                    f"Voucher: {order.voucher_code if order.voucher_code else 'None'}\n"
                    f"Date: {order.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Items ({len(order.items)}):"
                )
                
                for item in order.items:
                    lines.append(
                        f"  • {item.product_name} (Qty: {item.quantity}) - ${float(item.subtotal):.2f}"
                    )
                
                lines.append("")
            
            return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving orders: {str(e)}"
    finally:
        db.close()


def get_purchase_function() -> Dict[str, Any]:
    """
    Get OpenAI function definition for completing purchase.
    
    Returns:
        OpenAI function definition
    """
    return {
        "type": "function",
        "function": {
            "name": "purchase",
            "description": "Complete purchase with voucher code. Call ONCE when user provides voucher code. Call immediately without confirmation messages. Requires: items in cart, shipping info, and valid voucher code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "voucher_code": {
                        "type": "string",
                        "description": "The voucher code to use for payment"
                    }
                },
                "required": ["voucher_code"]
            }
        }
    }


def execute_purchase(session_id: str, voucher_code: str) -> str:
    """
    Execute purchase with voucher validation and shipping info check.
    
    Args:
        session_id: User session identifier
        voucher_code: Voucher code to use
        
    Returns:
        Purchase result message
    """
    db = SessionLocal()
    try:
        # Get cart
        cart = cart_manager.get_cart(session_id)
        if not cart:
            return "Error: Your cart is empty. Add items to your cart before purchasing."
        
        # Check for shipping information
        shipping_info = db.query(ShippingInfo).filter(
            ShippingInfo.session_id == session_id
        ).first()
        
        if not shipping_info:
            return (
                "Error: Shipping information is required before purchase. "
                "Please provide your full name, address, city, and zip code first."
            )
        
        cart_total = cart_manager.get_cart_total(session_id)
        
        # Validate voucher
        voucher = db.query(Voucher).filter(Voucher.code == voucher_code).first()
        if not voucher:
            return f"Error: Invalid voucher code '{voucher_code}'. Please check and try again."
        
        # IDEMPOTENCY CHECK: First check if an order already exists for this voucher_code
        # This handles the case where purchase is called multiple times
        existing_order = db.query(Order).filter(
            Order.voucher_code == voucher_code
        ).first()
        
        if existing_order:
            # Purchase already completed - return simple message (idempotent behavior)
            return f"✅ Your purchase has already been placed. Order ID: {existing_order.id}"
        
        # If no existing order, check if voucher is marked as used (shouldn't happen if order exists)
        if voucher.is_used:
            return f"Error: Voucher '{voucher_code}' has already been used."
        
        # Check if voucher amount is sufficient
        if float(voucher.amount) < cart_total:
            return (
                f"Error: Insufficient voucher balance. "
                f"Your cart total is ${cart_total:.2f}, but your voucher is worth ${float(voucher.amount):.2f}. "
                f"Please remove some items or use a voucher with sufficient balance."
            )
        
        # Create order
        order = Order(
            session_id=session_id,
            voucher_code=voucher_code,
            total_amount=cart_total,
            status="completed"
        )
        db.add(order)
        db.flush()  # Get order ID
        
        # Create order items
        for cart_item in cart:
            order_item = OrderItem(
                order_id=order.id,
                product_id=cart_item.product_id,
                product_name=cart_item.product_name,
                quantity=cart_item.quantity,
                unit_price=cart_item.unit_price,
                subtotal=cart_item.subtotal
            )
            db.add(order_item)
        
        # Mark voucher as used (once used, it cannot be reused even if order is less than voucher amount)
        voucher.is_used = True
        voucher.used_by_session = session_id
        voucher.used_at = datetime.now()
        
        # Commit transaction
        db.commit()
        
        # Refresh order to ensure items are loaded
        db.refresh(order)
        # Explicitly load items relationship
        order = db.query(Order).options(joinedload(Order.items)).filter(Order.id == order.id).first()
        
        # Clear cart
        cart_manager.clear_cart(session_id)
        
        # Build detailed success message
        items_summary = []
        for item in order.items:
            items_summary.append(f"  • {item.product_name} (Qty: {item.quantity}) - ${float(item.subtotal):.2f}")
        
        return (
            f"✅ Purchase completed successfully! Your order has been placed and saved.\n\n"
            f"Order Details:\n"
            f"  Order ID: {order.id}\n"
            f"  Status: {order.status}\n"
            f"  Total Amount: ${cart_total:.2f}\n"
            f"  Voucher Code Used: {voucher_code}\n"
            f"  Remaining Voucher Balance: ${float(voucher.amount) - cart_total:.2f}\n"
            f"\nOrder Items:\n" + "\n".join(items_summary) + "\n"
            f"\nShipping Address:\n"
            f"  {shipping_info.full_name}\n"
            f"  {shipping_info.address}\n"
            f"  {shipping_info.city}, {shipping_info.zip_code}\n"
            f"\nOrder Date: {order.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"\nThank you for your purchase! Your order is confirmed and will be processed shortly."
        )
    except Exception as e:
        db.rollback()
        return f"Error processing purchase: {str(e)}"
    finally:
        db.close()
