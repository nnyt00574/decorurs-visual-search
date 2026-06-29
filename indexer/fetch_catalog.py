"""
Pulls the full DecorUrs catalog via Shopify's public storefront JSON
endpoint (every Shopify store exposes this by default), paginates through
all results, and normalizes them into the fields the indexer needs:
product_id, name, image_url, product_url, price.

Run this any time the catalog changes, then re-run index_products.py.
"""

import json
import time
import requests

BASE_URL = "https://decorurs.com/collections/all/products.json"


def fetch_all_products() -> list[dict]:
    products = []
    page = 1
    while True:
        resp = requests.get(BASE_URL, params={"limit": 250, "page": page})
        resp.raise_for_status()
        batch = resp.json().get("products", [])
        if not batch:
            break
        products.extend(batch)
        print(f"Page {page}: fetched {len(batch)} products")
        page += 1
        time.sleep(0.5)  # be polite to the store
    return products


def normalize(products: list[dict]) -> list[dict]:
    items = []
    for p in products:
        images = p.get("images", [])
        variants = p.get("variants", [])
        if not images or not variants:
            continue
        items.append(
            {
                "product_id": p["id"],
                "name": p["title"],
                "image_urls": [img["src"] for img in images],
                "product_url": f"https://decorurs.com/products/{p['handle']}",
                "price": variants[0]["price"],
            }
        )
    return items


if __name__ == "__main__":
    raw = fetch_all_products()
    items = normalize(raw)
    with open("catalog.json", "w") as f:
        json.dump(items, f, indent=2)
    print(f"Saved {len(items)} products to catalog.json")
