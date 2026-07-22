import os
import json
import asyncio
import requests
from pinecone import Pinecone
from clip_service import ServerlessClipService

INDEX_NAME = "decorurs-products"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

async def index_catalog():
    print("Loading catalog...")
    try:
        with open("catalog.json", "r") as f:
            catalog = json.load(f)
    except FileNotFoundError:
        print("Error: catalog.json not found. Please run your Shopify fetcher script first.")
        return

    print(f"Connecting to Pinecone index '{INDEX_NAME}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Verify the index exists, if not, you'll need to create it in the Pinecone dashboard
    try:
        index = pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Could not connect to Pinecone index: {e}")
        return

    clip = ServerlessClipService.get()
    vectors_to_upsert = []
    
    # We need a unique string ID for every single image vector in Pinecone
    point_id_counter = 1

    print(f"Processing {len(catalog)} products...")
    
    for product in catalog:
        print(f"Indexing: {product['name']}")
        
        for image_url in product.get("image_urls", []):
            try:
                # 1. Download the image from Shopify
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
                image_bytes = resp.content

                # 2. Get vector (Hugging Face) and classification (OpenAI)
                analysis = await clip.analyze_image_from_bytes(image_bytes)
                
                # 3. Format the metadata for Pinecone filtering and frontend display
                metadata = {
                    "product_id": product["product_id"],
                    "name": product["name"],
                    "image_url": image_url,
                    "product_url": product["product_url"],
                    "price": str(product["price"]), # Pinecone metadata prefers strings/numbers
                    "material": analysis["material"],
                    "shape": analysis["shape"]
                }

                # 4. Append to our batch
                vectors_to_upsert.append({
                    "id": f"img_{point_id_counter}",
                    "values": analysis["vector"],
                    "metadata": metadata
                })
                
                point_id_counter += 1

                # 5. Batch upsert every 50 vectors (saves network requests)
                if len(vectors_to_upsert) >= 50:
                    print(f"Upserting batch of {len(vectors_to_upsert)} vectors to Pinecone...")
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
                    
            except Exception as e:
                print(f"Failed to process image {image_url}: {e}")

    # Upsert any remaining vectors left in the list
    if vectors_to_upsert:
        print(f"Upserting final batch of {len(vectors_to_upsert)} vectors to Pinecone...")
        index.upsert(vectors=vectors_to_upsert)

    print(f"Indexing complete! Successfully processed {point_id_counter - 1} total images.")

if __name__ == "__main__":
    # Because our ServerlessClipService relies on async calls to OpenAI, 
    # we run the indexer inside an asyncio event loop.
    asyncio.run(index_catalog())