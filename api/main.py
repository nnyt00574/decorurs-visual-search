"""
DecorUrs Visual Search API.

POST /search with an image file -> returns the top 10 visually similar
products from the indexed catalog, ranked by cosine similarity with a
small boost for products whose predicted material matches the upload's.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from clip_service import ClipService

COLLECTION_NAME = "decorurs_products"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png"}

# How many raw points to pull from Qdrant before deduping down to
# products. Needs to comfortably exceed (products we want) x (images per
# product) so a product with several indexed photos doesn't crowd out
# other genuinely relevant products in the candidate pool.
CANDIDATE_POOL_SIZE = 50
RESULTS_TO_RETURN = 10

# Small additive nudge applied when a candidate's material matches the
# upload's predicted material. Kept small relative to the typical spread
# of cosine similarities (usually 0.6-0.95) so it re-orders close ties
# without overriding a clearly stronger visual match.
MATERIAL_BOOST = 0.04

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")


class ProductResult(BaseModel):
    product_id: int
    name: str
    image_url: str
    product_url: str
    price: str
    material: str
    score: float


class SearchResponse(BaseModel):
    results: list[ProductResult]
    query_material: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.clip = ClipService.get()
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL)
    yield
    await app.state.qdrant.close()


app = FastAPI(title="DecorUrs Visual Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    try:
        # CLIP inference (plus background-removal cropping) is CPU-bound
        # and synchronous -- run it off the event loop.
        analysis = await run_in_threadpool(app.state.clip.analyze_image_from_bytes, data)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image")

    query_material = analysis["material"]

    result = await app.state.qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=analysis["vector"],
        limit=CANDIDATE_POOL_SIZE,
    )

    # Multiple points can belong to the same product (one per catalog
    # image) -- keep only the best-scoring point per product before
    # ranking, so one heavily-photographed product can't crowd out others.
    best_by_product = {}
    for point in result.points:
        pid = point.payload["product_id"]
        if pid not in best_by_product or point.score > best_by_product[pid].score:
            best_by_product[pid] = point

    def ranking_key(point):
        boost = MATERIAL_BOOST if point.payload.get("material") == query_material else 0.0
        return point.score + boost

    ranked = sorted(best_by_product.values(), key=ranking_key, reverse=True)
    top = ranked[:RESULTS_TO_RETURN]

    return SearchResponse(
        query_material=query_material,
        results=[
            ProductResult(
                product_id=point.payload["product_id"],
                name=point.payload["name"],
                image_url=point.payload["image_url"],
                product_url=point.payload["product_url"],
                price=point.payload["price"],
                material=point.payload["material"],
                score=round(ranking_key(point), 4),
            )
            for point in top
        ],
    )
