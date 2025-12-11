"""Script to import products from products.json into the database."""
import json
import httpx
from typing import List, Dict, Any


def load_products(file_path: str) -> List[Dict[str, Any]]:
    """Load products from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_product(api_url: str, product: Dict[str, Any]) -> bool:
    """Create a single product via API."""
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{api_url}/admin/products/",
                json=product,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            if response.status_code == 201:
                print(f"✓ Created: {product['name']} ({product['sku']})")
                return True
            else:
                print(f"✗ Failed: {product['name']} ({product['sku']}) - {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"  Error: {error_detail}")
                except:
                    print(f"  Error: {response.text}")
                return False
    except Exception as e:
        print(f"✗ Error creating {product['name']}: {str(e)}")
        return False


def main():
    """Main function to import products (remaining after first 30)."""
    # Update this to your API URL
    api_url = "http://localhost:8000"
    
    # Load products from JSON file
    all_products = load_products("products.json")
    
    # Import products starting from index 30 (skip first 30)
    # products = all_products[30:]
    
    print(f"Found {len(all_products)} products in file, importing remaining {len(products)} (products 31-{len(all_products)})...")
    print("-" * 60)
    
    success_count = 0
    failed_count = 0
    
    for product in products:
        if create_product(api_url, product):
            success_count += 1
        else:
            failed_count += 1
    
    print("-" * 60)
    print(f"Import complete: {success_count} succeeded, {failed_count} failed")


if __name__ == "__main__":
    main()

