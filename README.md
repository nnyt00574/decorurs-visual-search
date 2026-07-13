# DecorUrs Visual Search

An AI-powered visual search engine for the DecorUrs furniture catalog. This system allows users to upload an image of a piece of furniture and instantly find visually similar products from the Shopify store, filtered by predicted material and shape.

## Architecture

The project consists of three main components running via Docker Compose:

1. **Indexer (`/indexer`)**: A batch job that fetches the entire product catalog from the Shopify API, processes the images, and pushes them into the vector database.
2. **Search API (`/api`)**: A FastAPI backend that accepts user image uploads, processes them identically to the indexer, and queries the vector database for the nearest neighbors.
3. **Qdrant (`qdrant`)**: The vector database used to store and query the high-dimensional image embeddings.

## AI & Computer Vision Pipeline

Every image (both catalog product shots and user uploads) goes through a strict pipeline in `clip_service.py`:

* **Background Removal (`rembg`)**: Isolates the furniture from lifestyle backgrounds/rooms so the embedding focuses entirely on the product, not the floor or walls.
* **CLIP Embeddings (OpenCLIP ViT-L-14)**: Generates a 768-dimensional visual feature vector for similarity search. We use a Large model for better texture/material differentiation (e.g., distinguishing wood grain from marble veining).
* **Zero-Shot Classification**: Automatically labels the product's `material` and `shape` using CLIP's text-image similarity space, eliminating the need for a separate classification model. 
  * *Prompt Ensembling*: Averages multiple text prompts (e.g., "a round table", "a table with a round top") to neutralize semantic bias in CLIP's text space.
  * *Aspect-Ratio Vetoing*: Uses the bounding box geometry from the background removal step to veto impossible shape predictions (e.g., preventing an elongated console table from being classified as "round").

## Running Locally

**1. Start the infrastructure:**
\`\`\`bash
docker compose up -d
\`\`\`

**2. Index the catalog:**
\`\`\`bash
docker compose run --rm indexer
\`\`\`

**3. Search via API:**
\`\`\`bash
curl -X POST http://localhost:8000/search -F "file=@/path/to/your/image.jpg"
\`\`\`
