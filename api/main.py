"""
DecorUrs Visual Search API.

POST /search with an image file -> returns up to 10 visually similar
products from the indexed catalog that also match the upload's predicted
material AND tabletop shape (e.g. only rectangular, only solid wood).
Material/shape are hard filters, not just a ranking nudge -- a round
coffee table should never show up as a match for a rectangular wood
dining table, no matter how visually similar the wood grain looks.
If nothing in the catalog matches both, `results` comes back empty and
the frontend shows a "no matching tables" message with a custom-order
contact.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

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

# Shape is classified against only 4 known categories (rectangular,
# square, round, oval). An upload that doesn't actually look like any of
# them (a star-shaped table, a live-edge slab, etc.) still gets forced
# into whichever is *closest* -- but with low confidence. Rather than
# silently treat that low-confidence guess as a real match, we require a
# minimum confidence before filtering on it; below this, we treat the
# shape as unrecognized and skip straight to "no matching tables" instead
# of returning products of the nearest-guessed shape.
SHAPE_CONFIDENCE_THRESHOLD = 0.45

# Same idea for material -- 8 categories, so a confident classification
# should clear this comfortably; a genuinely ambiguous material (e.g. a
# painted or mixed-material piece) shouldn't be force-matched either.
MATERIAL_CONFIDENCE_THRESHOLD = 0.35

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")


class ProductResult(BaseModel):
    product_id: int
    name: str
    image_url: str
    product_url: str
    price: str
    material: str
    shape: str
    score: float


class SearchResponse(BaseModel):
    results: list[ProductResult]
    query_material: str
    query_shape: str


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
    query_shape = analysis["shape"]

    # If the model isn't actually confident the upload is one of our
    # known shapes/materials, don't force it into the nearest bucket and
    # search on that guess -- that's how a star-shaped table ends up
    # matched against round tables. Treat it as "we don't carry this" and
    # let the frontend show the custom-order message instead.
    if (
        analysis["shape_confidence"] < SHAPE_CONFIDENCE_THRESHOLD
        or analysis["material_confidence"] < MATERIAL_CONFIDENCE_THRESHOLD
    ):
        return SearchResponse(query_material=query_material, query_shape=query_shape, results=[])

    # Hard filter: only points whose predicted material AND shape match
    # the upload. This runs inside Qdrant (not as a post-hoc re-rank), so
    # a round table or a table in the wrong material is excluded from the
    # candidate pool entirely -- it can never appear in results just
    # because it happened to be visually similar overall.
    match_filter = Filter(
        must=[
            FieldCondition(key="material", match=MatchValue(value=query_material)),
            FieldCondition(key="shape", match=MatchValue(value=query_shape)),
        ]
    )

    result = await app.state.qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=analysis["vector"],
        query_filter=match_filter,
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

    ranked = sorted(best_by_product.values(), key=lambda p: p.score, reverse=True)
    top = ranked[:RESULTS_TO_RETURN]

    return SearchResponse(
        query_material=query_material,
        query_shape=query_shape,
        results=[
            ProductResult(
                product_id=point.payload["product_id"],
                name=point.payload["name"],
                image_url=point.payload["image_url"],
                product_url=point.payload["product_url"],
                price=point.payload["price"],
                material=point.payload["material"],
                shape=point.payload["shape"],
                score=round(point.score, 4),
            )
            for point in top
        ],
    )
# cache test
