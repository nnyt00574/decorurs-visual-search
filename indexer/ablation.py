"""
Ablation study for the visual search pipeline.

Since there's no separate "customer photo" dataset, this uses a
leave-one-out evaluation on the catalog itself: for every product with
2+ photos, one photo is held out as a synthetic "query" and the rest
stay in the searchable gallery (other products' photos are also in the
gallery, as distractors). The question each config has to answer: does
the held-out photo retrieve its own product back?

Products with only one photo can't generate a valid held-out query (the
only photo they have would need to be in both query and gallery), so
they're query-ineligible but still sit in the gallery as distractors --
exactly the role they'd play in real search traffic.

Metrics reported per config:
  Recall@1 / Recall@5 / Recall@10  -- did the correct product appear
                                       in the top K results?
  MRR (Mean Reciprocal Rank)        -- rewards ranking it higher, not
                                       just "somewhere in top 10"
  Avg query embed time (s)          -- the part of search latency this
                                       script can actually measure

Run with the indexer's existing image -- no new dependencies, no rebuild:
  docker compose run --rm indexer uv run python ablation.py

Results print to stdout. To also persist them to a file on the host,
mount a volume for this one-off run:
  docker compose run --rm -v "$(pwd)/eval-results:/app/eval-results" \\
      indexer uv run python ablation.py
"""

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open_clip
import requests
import torch
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm

CATALOG_PATH = Path("catalog.json")
IMAGE_CACHE_DIR = Path("image_cache")
RESULTS_DIR = Path("eval-results")
REMBG_MODEL = "u2netp"

MATERIAL_LABELS = [
    "marble",
    "travertine stone",
    "granite stone",
    "solid wood",
    "metal",
    "glass",
    "rattan or wicker",
    "upholstered fabric",
]
MATERIAL_BOOST = 0.04  # matches api/main.py, so the ablation reflects production behavior

# --- Edit this list to add/remove configs. Each is a full retrieval run. ---
# ViT-H-14 has a much bigger checkpoint (~2.5GB) and is noticeably slower
# on CPU -- comment it out below for a quicker pass if you just want the
# crop / material-boost comparisons.
CONFIGS = [
    {"name": "ViT-B-32, no crop",              "model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", "crop": False, "material_boost": False},
    {"name": "ViT-B-32, crop",                 "model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", "crop": True,  "material_boost": False},
    {"name": "ViT-L-14, no crop",              "model": "ViT-L-14", "pretrained": "laion2b_s32b_b82k", "crop": False, "material_boost": False},
    {"name": "ViT-L-14, crop",                 "model": "ViT-L-14", "pretrained": "laion2b_s32b_b82k", "crop": True,  "material_boost": False},
    {"name": "ViT-L-14, crop + material boost","model": "ViT-L-14", "pretrained": "laion2b_s32b_b82k", "crop": True,  "material_boost": True},
    {"name": "ViT-H-14, crop",                 "model": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "crop": True,  "material_boost": False},
]


def load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        raise SystemExit(
            "catalog.json not found. Run fetch_catalog.py first, or run "
            "this from a container that already has it (the indexer's "
            "default entrypoint produces it)."
        )
    with open(CATALOG_PATH) as f:
        return json.load(f)


def cached_image(url: str) -> Image.Image:
    """Downloads an image once and reuses the local copy across every
    config -- otherwise each of N configs re-fetches every image from
    the live DecorUrs site, which is both slow and rude."""
    IMAGE_CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()
    path = IMAGE_CACHE_DIR / f"{key}.jpg"
    if not path.exists():
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return Image.open(path).convert("RGB")


def build_eval_split(catalog: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (gallery, queries). Every image is in the gallery.
    Only one image per multi-image product is also used as a query."""
    gallery, queries = [], []
    eligible_products = 0
    for item in catalog:
        urls = item["image_urls"]
        for url in urls:
            gallery.append({"product_id": item["product_id"], "name": item["name"], "image_url": url})
        if len(urls) >= 2:
            eligible_products += 1
            queries.append({"product_id": item["product_id"], "name": item["name"], "image_url": urls[0]})

    if eligible_products == 0:
        raise SystemExit(
            "No product in the catalog has 2+ photos, so there's no valid "
            "held-out query to test with (the only photo a single-image "
            "product has would have to be excluded from its own gallery "
            "as 'self', leaving nothing left to find). Add more photos per "
            "product, or adapt this script to use real customer photos as "
            "queries instead of catalog leave-one-out."
        )

    return gallery, queries


def load_model(model_name: str, pretrained: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def embed_material_labels(model, tokenizer, device) -> torch.Tensor:
    prompts = [f"a furniture piece made of {label}" for label in MATERIAL_LABELS]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features


def crop_to_subject(image: Image.Image, rembg_session, padding_frac: float = 0.04) -> Image.Image:
    try:
        rgba = remove(image, session=rembg_session)
    except Exception:
        return image
    alpha = np.array(rgba.split()[-1])
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0 or len(ys) == 0:
        return image
    w, h = image.size
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x, pad_y = int((x1 - x0) * padding_frac), int((y1 - y0) * padding_frac)
    return image.crop((max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y)))


def embed_one(image: Image.Image, model, preprocess, device, crop: bool, rembg_session) -> np.ndarray:
    if crop:
        image = crop_to_subject(image, rembg_session)
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features /= features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).cpu().numpy()


def classify_material(image_features: np.ndarray, material_text_features: torch.Tensor) -> str:
    sims = material_text_features.cpu().numpy() @ image_features
    return MATERIAL_LABELS[int(sims.argmax())]


def run_config(config: dict, gallery: list[dict], queries: list[dict], rembg_session) -> dict:
    model, preprocess, tokenizer, device = load_model(config["model"], config["pretrained"])
    material_text_features = embed_material_labels(model, tokenizer, device) if config["material_boost"] else None

    # Embed the whole gallery once for this config.
    gallery_vectors, gallery_materials = [], []
    for item in tqdm(gallery, desc=f"[{config['name']}] embedding gallery", leave=False):
        img = cached_image(item["image_url"])
        vec = embed_one(img, model, preprocess, device, config["crop"], rembg_session)
        gallery_vectors.append(vec)
        if config["material_boost"]:
            gallery_materials.append(classify_material(vec, material_text_features))
        else:
            gallery_materials.append(None)
    gallery_matrix = np.stack(gallery_vectors)  # [N, D]
    gallery_pids = [item["product_id"] for item in gallery]
    gallery_urls = [item["image_url"] for item in gallery]

    ranks = []
    embed_times = []
    for q in tqdm(queries, desc=f"[{config['name']}] querying", leave=False):
        img = cached_image(q["image_url"])
        t0 = time.perf_counter()
        qvec = embed_one(img, model, preprocess, device, config["crop"], rembg_session)
        embed_times.append(time.perf_counter() - t0)

        scores = gallery_matrix @ qvec
        if config["material_boost"]:
            qmat = classify_material(qvec, material_text_features)
            scores = scores + MATERIAL_BOOST * np.array([1.0 if m == qmat else 0.0 for m in gallery_materials])

        order = np.argsort(-scores)
        # Find the rank of the correct product, skipping the literal same
        # photo (its own image_url) -- otherwise every config trivially
        # "finds itself" at rank 1 regardless of model quality, which
        # would make every number below meaningless.
        rank = None
        position = 0
        for idx in order:
            if gallery_urls[idx] == q["image_url"]:
                continue
            position += 1
            if gallery_pids[idx] == q["product_id"]:
                rank = position
                break
        ranks.append(rank if rank is not None else len(gallery_pids))

    ranks = np.array(ranks)
    return {
        "name": config["name"],
        "n_queries": len(queries),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "avg_embed_s": float(np.mean(embed_times)),
    }


def main():
    catalog = load_catalog()
    gallery, queries = build_eval_split(catalog)
    print(f"Gallery: {len(gallery)} images. Queries: {len(queries)} (held-out, 2+ photo products only).\n")

    rembg_session = new_session(REMBG_MODEL)

    results = []
    for config in CONFIGS:
        print(f"=== Running: {config['name']} ===")
        results.append(run_config(config, gallery, queries, rembg_session))

    headers = ["name", "n_queries", "recall@1", "recall@5", "recall@10", "mrr", "avg_embed_s"]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    header_row = " | ".join(h.ljust(14) for h in headers)
    print(header_row)
    print("-" * len(header_row))
    for r in results:
        row = " | ".join(
            (f"{r[h]:.3f}" if isinstance(r[h], float) else str(r[h])).ljust(14) for h in headers
        )
        print(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAlso wrote {csv_path} (persists only if you mounted eval-results/ as a volume).")


if __name__ == "__main__":
    main()
