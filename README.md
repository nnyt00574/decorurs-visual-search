# DecorUrs Visual Search

Upload a furniture photo, get back the 10 most visually similar products
from the DecorUrs catalog.

Fully containerized: Qdrant, the search API, and the frontend all run in
Docker. Catalog indexing runs as a one-off job in the same Compose stack.

## Stack

| Layer       | Tech                                              |
|-------------|---------------------------------------------------|
| Vector DB   | Qdrant v1.17 (server mode, persistent volume)     |
| Embeddings  | OpenCLIP ViT-L-14 (laion2b)                       |
| Preprocess  | rembg background removal + crop-to-subject        |
| Matching    | cosine similarity, hard-filtered by CLIP zero-shot material + shape |
| API         | FastAPI 0.137 (async, lifespan, Pydantic models)  |
| Frontend    | Next.js 16.2 / React 19 (App Router, Turbopack)   |
| Python deps | uv + pyproject.toml                               |

## How matching works now

1. **Every** product image is embedded, not just a listing's first photo
   — a customer's photo might match a side angle or detail shot.
2. Before embedding, both catalog images and the uploaded photo go
   through background removal and get cropped to the furniture subject,
   so room context (floors, walls, other furniture) doesn't dilute the
   match.
3. Each image is also run through CLIP zero-shot classification against
   a fixed material vocabulary (marble, travertine, wood, metal, glass,
   etc.) and a fixed shape vocabulary (rectangular, square, round, oval)
   — no separate classifiers, just comparing the same image embedding
   against material/shape text prompts in CLIP's shared space.
4. At search time: query Qdrant for the top 50 candidate images that
   *also* match the upload's predicted material AND shape (a hard
   filter applied inside Qdrant, not a post-hoc re-rank), collapse down
   to one (best-scoring) point per product, then return the top 10 by
   cosine similarity. A round table or a table in a different material
   is excluded from the candidate pool entirely — it can't surface just
   because it happens to look visually similar overall.
5. If no catalog product matches both the material and shape, `results`
   comes back empty and the UI shows a "no matching tables" message with
   a custom-order contact.

The detected material and shape are returned in the API response
(`query_material`, `query_shape`, plus `material`/`shape` fields per
result) and shown in the UI.

## Ablation study

`indexer/ablation.py` compares CLIP model size, cropping, and the
material boost on actual retrieval quality, instead of judging changes
by eyeballing results. It reuses the indexer's already-built image, so
no rebuild is needed:

```bash
docker compose run --rm indexer uv run python ablation.py
```

Methodology: since there's no separate customer-photo dataset, it does
leave-one-out evaluation on the catalog itself -- for every product with
2+ photos, one photo is held out as a query (excluded from its own
gallery) and the rest stay searchable as the gallery, alongside every
other product's photos as distractors. Reports Recall@1/5/10 and MRR
per config.

**Note:** if most products in the catalog only have one photo, the
script will exit with a clear message rather than print misleading
numbers -- a meaningful eval genuinely needs 2+ photos per product to
test whether a different photo of the same item gets found, not just
whether a photo finds an identical copy of itself.

Edit the `CONFIGS` list at the top of the script to add/remove
model/crop/material-boost combinations. `ViT-H-14` is included but has
a ~2.5GB checkpoint and is noticeably slower on CPU -- comment it out
for a quicker pass.

## Project structure

```
decorurs-visual-search/
├── docker-compose.yml
├── .env.example
├── indexer/        # one-off job: catalog -> embeddings -> Qdrant
├── api/            # FastAPI search service
└── frontend/       # Next.js upload + results page
```

## Prerequisites

- Docker Desktop (or Docker Engine + Compose v2) running
- That's it — Python, Node, and all dependencies live inside the containers.

## 1. Configure (optional)

```bash
cp .env.example .env
```

Defaults work for local use as-is. Only edit `.env` if you're deploying
somewhere other than `localhost`.

## 2. Start Qdrant, the API, and the frontend

```bash
docker compose up -d --build
```

This starts three services:
- `qdrant` — vector database, port 6333
- `api` — FastAPI search endpoint, port 8000
- `frontend` — Next.js app, port 3000

Check everything's healthy:
```bash
docker compose ps
curl http://localhost:8000/health   # {"status":"ok"}
```

## 3. Index the catalog (one-off job)

The catalog won't search anything until it's indexed. This runs as a
separate job, not as a long-running service:

```bash
docker compose run --rm indexer
```

Model weights are now baked into the image at build time (see Dockerfile),
so this won't re-download anything -- it goes straight to embedding. It
embeds every image of every product, classifies each image's material
and shape, and upserts one Qdrant point per image. Ends with something
like `Indexed 180 images across 63 products. Failed: 0`. Re-run any time
the catalog changes.

**Note:** shape was added after the original index was built. If you're
upgrading an existing deployment, re-run this step once so every point
gets a `shape` payload field — otherwise the API's material+shape filter
will exclude everything indexed before this change.

## 4. Use it

Open **http://localhost:3000**, upload a furniture photo, and you should
see 10 ranked results within a couple of seconds.

## Useful commands

```bash
docker compose logs -f api        # tail API logs
docker compose restart api        # restart just the API
docker compose down                # stop everything (keeps the qdrant_data volume)
docker compose down -v             # stop everything AND delete indexed data
```

## Notes on the modernization

- **Qdrant client API**: `recreate_collection()` and `.search()` are
  deprecated in current `qdrant-client`. This project uses
  `collection_exists()` / `create_collection()` and `query_points()`
  instead — see `indexer/index_products.py` and `api/main.py`.
- **Async all the way down in the API**: `AsyncQdrantClient` for the
  vector search, and the CPU-bound CLIP embedding call is offloaded via
  `run_in_threadpool` so it doesn't block the event loop under load.
- **uv instead of pip+venv**: each Python service has a `pyproject.toml`;
  Docker builds install dependencies with `uv sync`. No `requirements.txt`,
  no manual virtualenv activation.
- **Next.js standalone output**: `frontend/next.config.js` sets
  `output: "standalone"`, so the production Docker image only ships the
  files actually needed to run (`node server.js`), not the full
  `node_modules` tree.
- **`NEXT_PUBLIC_API_URL` is baked in at build time**, not read at
  container runtime — that's how Next.js env vars work for anything used
  in client-side code. It's passed as a Docker build arg from
  `docker-compose.yml`, sourced from `.env`.
- **Qdrant storage uses a named volume**, not a bind mount — Qdrant's own
  docs flag bind mounts (especially on Docker Desktop/WSL) as prone to
  silent storage corruption.
- **Model weights are baked into the Docker image at build time**
  (`warm_models.py`, run during `docker build`), not downloaded on first
  request/run. Means larger images, but predictable container startup and
  no repeated downloads on restart.
- Pinned versions as of this build: `qdrant/qdrant:v1.17.1`,
  `qdrant-client==1.18.0`, `fastapi[standard]==0.137.2`, `next@^16.2.0`,
  `react@^19.2.0`, OpenCLIP `ViT-L-14` (`laion2b_s32b_b82k`). Check for
  newer releases before treating these as permanent.

## If you'd rather run it without Docker

Each service still works standalone with `uv` (Python) and `npm` (frontend)
pointed at a Qdrant instance of your choosing — just set `QDRANT_URL` and
`NEXT_PUBLIC_API_URL` accordingly. Ask if you want that walkthrough instead.
