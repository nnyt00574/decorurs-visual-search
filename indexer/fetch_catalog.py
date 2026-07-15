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

# Some storefronts rate-limit or flag the default python-requests
# User-Agent more aggressively than a normal browser one.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# The storefront endpoint is unauthenticated and can rate-limit bursts of
# requests (e.g. re-running the indexer a few times in a row while
# iterating locally). A single 429 shouldn't kill the whole job -- retry
# with backoff first, respecting the server's own Retry-After header when
# it sends one. Budget is generous (up to ~10.5 min cumulative) because a
# 120s wait was observed to not be enough to clear an active block.
MAX_RETRIES = 6
BACKOFF_SECONDS = 10  # doubled on each subsequent retry: 10,20,40,80,160,320


def _get_with_retry(url: str, params: dict) -> requests.Response:
    resp = None
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, headers=HEADERS)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = int(resp.headers.get("Retry-After", BACKOFF_SECONDS * (2 ** attempt)))
        print(f"Rate limited (429), attempt {attempt + 1}/{MAX_RETRIES} -- waiting {wait}s")
        time.sleep(wait)
    resp.raise_for_status()  # retries exhausted: surface the last response's error
    return resp


def fetch_all_products() -> list[dict]:
    products = []
    page = 1
    while True:
        resp = _get_with_retry(BASE_URL, {"limit": 250, "page": page})
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