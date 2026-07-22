import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone

from clip_service import ServerlessClipService

INDEX_NAME = "decorurs-products"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png"}
CANDIDATE_POOL_SIZE = 50
RESULTS_TO_RETURN = 10

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "https://decorurs.com").split(",")

class ProductResult(BaseModel):
    product_id: intpinecone login
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
    # Initialize Serverless Services
    app.state.clip = ServerlessClipService.get()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    app.state.pinecone_index = pc.Index(INDEX_NAME)
    yield
    # No local DB connections to close in serverless

app = FastAPI(title="DecorUrs Visual Search API (Serverless)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "architecture": "serverless"}

@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    try:
        # All heavy ML is now handled via async external APIs
        analysis = await app.state.clip.analyze_image_from_bytes(data)
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="Could not process image via ML APIs")

    query_material = analysis["material"]
    query_shape = analysis["shape"]

    # Fallback if OpenAI couldn't confidently classify it as a table
    if query_shape == "unknown" or query_material == "unknown":
        return SearchResponse(query_material=query_material, query_shape=query_shape, results=[])

    # Pinecone Metadata Filtering (replaces Qdrant FieldCondition)
    match_filter = {
        "material": {"$eq": query_material},
        "shape": {"$eq": query_shape}
    }

    # Query Pinecone Index
    result = app.state.pinecone_index.query(
        vector=analysis["vector"],
        filter=match_filter,
        top_k=CANDIDATE_POOL_SIZE,
        include_metadata=True
    )

    # Deduplicate by product_id (keeping highest score)
    best_by_product = {}
    for match in result.matches:
        pid = match.metadata["product_id"]
        if pid not in best_by_product or match.score > best_by_product[pid].score:
            best_by_product[pid] = match

    ranked = sorted(best_by_product.values(), key=lambda m: m.score, reverse=True)
    top = ranked[:RESULTS_TO_RETURN]

    return SearchResponse(
        query_material=query_material,
        query_shape=query_shape,
        results=[
            ProductResult(
                product_id=int(match.metadata["product_id"]),
                name=match.metadata["name"],
                image_url=match.metadata["image_url"],
                product_url=match.metadata["product_url"],
                price=str(match.metadata["price"]),
                material=match.metadata["material"],
                shape=match.metadata["shape"],
                score=round(match.score, 4),
            )
            for match in top
        ],
    )