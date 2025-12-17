"""Tools for agents."""
from .retrieval import (
    get_handbook_retrieval_function,
    execute_handbook_retrieval,
    get_product_search_function,
    execute_product_search
)
from .order import (
        get_add_to_cart_function,
        get_edit_item_in_cart_function,
        get_remove_from_cart_function,
        get_view_cart_function,
        get_shipping_info_function,
        get_create_shipping_info_function,
        get_purchase_function,
        execute_add_to_cart,
        execute_edit_item_in_cart,
        execute_remove_from_cart,
        execute_view_cart,
        execute_get_shipping_info,
        execute_create_shipping_info,
        execute_purchase
    )

__all__ = [
    "get_handbook_retrieval_function",
    "execute_handbook_retrieval",
    "get_product_search_function",
    "execute_product_search",
    "get_add_to_cart_function",
    "get_edit_item_in_cart_function",
    "get_remove_from_cart_function",
    "get_view_cart_function",
    "get_shipping_info_function",
    "get_create_shipping_info_function",
    "get_purchase_function",
    "execute_add_to_cart",
    "execute_edit_item_in_cart",
    "execute_remove_from_cart",
    "execute_view_cart",
    "execute_get_shipping_info",
    "execute_create_shipping_info",
    "execute_purchase"
]
