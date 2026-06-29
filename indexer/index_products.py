"""
Reads catalog.json (produced by fetch_catalog.py), embeds EVERY image of
EVERY product -- not just the first -- and upserts one Qdrant point per
image. A customer's photo might match a side-angle or detail shot rather
than a listing's primary image, so all of them need to be searchable.

Each point also carries a predicted material label so the API can boost
results whose material matches the uploaded photo's, not just its overall
visual similarity.

Safe to re-run from scratch -- it recreates the collection each time,
which is fine at this catalog size.
"""

import json
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

from clip_service import ClipService, VECTOR_SIZE

COLLECTION_NAME = "decorurs_products"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def main():
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    with open("catalog.json") as f:
        catalog = json.load(f)

    clip = ClipService.get()
    points, failed = [], []

    total_images = sum(len(item["image_urls"]) for item in catalog)
    progress = tqdm(total=total_images, desc="Embedding product images")

    for item in catalog:
        for image_url in item["image_urls"]:
            try:
                analysis = clip.analyze_image_from_url(image_url)
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, image_url))
                payload = {
                    "product_id": item["product_id"],
                    "name": item["name"],
                    "image_url": image_url,
                    "product_url": item["product_url"],
                    "price": item["price"],
                    "material": analysis["material"],
                }
                points.append(PointStruct(id=point_id, vector=analysis["vector"], payload=payload))
            except Exception as e:
                failed.append((item["name"], image_url, str(e)))
            progress.update(1)

    progress.close()

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Indexed {len(points)} images across {len(catalog)} products. Failed: {len(failed)}")
    for name, url, err in failed:
        print(f"  FAILED: {name} ({url}) -> {err}")


if __name__ == "__main__":
    main()
