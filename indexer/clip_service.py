"""
Loads an OpenCLIP model once and exposes helpers to turn furniture photos
into (a) a similarity-search embedding and (b) a predicted material label.
Used identically by the indexer (catalog images) and the API (a user's
upload) so both sides are processed the exact same way.

Two things happen to every image before it's embedded:
1. Background removal + crop to the foreground bounding box, so room
   context (floors, walls, other furniture in a customer's photo) doesn't
   dilute the embedding. Catalog product shots get the same treatment for
   consistency, even though most are already on a clean background.
2. Material and shape classification via CLIP zero-shot: the image
   embedding is compared against fixed sets of material/shape text
   prompts in the same CLIP space, no separate classifier needed.

Shape classification specifically combines two independent safeguards,
since they catch different failure modes:
- Prompt ensembling (see SHAPE_PROMPT_TEMPLATES / _embed_shape_labels):
  fixes a per-phrase bias in CLIP's zero-shot text embeddings themselves
  ("a table with a round top" sat closer to typical furniture photos than
  the other three phrasings on this checkpoint, regardless of the image),
  which was over-representing "round" across the whole catalog.
- Aspect-ratio veto (see ELONGATED_ASPECT_RATIO): fixes a separate,
  image-side failure mode where rembg's foreground box comes out clearly
  elongated (e.g. lifestyle shots where the box is closer to isolating
  just the table), which a truly round/square top should never produce.
"""

import torch
import open_clip
from PIL import Image
import requests
import numpy as np
from io import BytesIO
from rembg import remove, new_session

# ViT-L-14 instead of ViT-B-32: noticeably better at distinguishing fine
# material/texture differences (stone veining, wood grain, base style),
# which matters a lot in a catalog full of similarly-shaped tables in
# different finishes. Slower than B-32, but still well under a second
# per image on CPU -- and indexing is a one-off batch job anyway.
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
VECTOR_SIZE = 768

# Fixed material vocabulary classified via CLIP zero-shot, applied
# identically to catalog images and the user's upload. Kept short and
# mutually distinguishable rather than exhaustive -- a long, overlapping
# label list makes zero-shot classification noisier, not more accurate.
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

# Tabletop shape, classified the same zero-shot way as material. This is
# what lets the API tell a rectangular dining table apart from a round
# table or a square one -- without it, search only knows "looks similar
# overall" and happily mixes shapes together.
SHAPE_LABELS = [
    "rectangular",
    "square",
    "round",
    "oval",
]

# Multiple differently-worded prompts per shape label, averaged together
# in _embed_shape_labels (prompt ensembling). A single fixed phrasing has
# its own quirks in CLIP's text-embedding space independent of the image
# -- on this checkpoint, "a table with a round top" alone sat closer to
# typical furniture photos than the other three labels' single phrasings,
# which was silently over-representing "round" across the whole catalog
# regardless of actual tabletop shape. Averaging several phrasings per
# label cancels out any one phrasing's idiosyncratic position, so the
# resulting label vector reflects the shape concept itself.
SHAPE_PROMPT_TEMPLATES = [
    "a table with a {shape} top",
    "a {shape} table",
    "a {shape} shaped table top",
    "a photo of a {shape} table",
    "a dining table with a {shape} tabletop",
    "an overhead view of a {shape} table top",
    "furniture: a {shape} table",
]

# Foreground bounding-box width:height ratio beyond which a tabletop is
# too elongated to plausibly be round or square in typical product or
# customer photography. Used as a sanity check on CLIP's zero-shot shape
# guess -- CLIP has been observed confidently calling clearly elongated
# tables (long console tables, slab dining tables) "round", which a
# symmetric shape should never produce that silhouette for. This is a
# separate safeguard from prompt ensembling above: it catches image-side
# problems (e.g. rembg's foreground box including chairs/décor around the
# table), not the text-prompt bias.
ELONGATED_ASPECT_RATIO = 2.0

# Lightweight background-removal model (~4MB) -- enough to get a usable
# foreground bounding box without the size/latency cost of the full model.
REMBG_MODEL = "u2netp"


class ClipService:
    """Singleton wrapper so the (relatively heavy) model and session are
    loaded once per process rather than once per request or per image."""

    _instance = None

    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._rembg_session = new_session(REMBG_MODEL)
        self._material_text_features = self._embed_material_labels()
        self._shape_text_features = self._embed_shape_labels()

    @classmethod
    def get(cls) -> "ClipService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _embed_material_labels(self) -> torch.Tensor:
        prompts = [f"a furniture piece made of {label}" for label in MATERIAL_LABELS]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def _embed_shape_labels(self) -> torch.Tensor:
        """Prompt-ensembled shape label embeddings: for each shape, embed
        several differently-worded prompts, average the (individually
        normalized) embeddings, then re-normalize. This removes the bias
        of any single phrasing dominating a label's position in CLIP's
        text-embedding space. Output shape is unchanged from the
        single-prompt version ([len(SHAPE_LABELS), VECTOR_SIZE]), so
        nothing downstream needs to change."""
        label_features = []
        for label in SHAPE_LABELS:
            prompts = [t.format(shape=label) for t in SHAPE_PROMPT_TEMPLATES]
            tokens = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                mean_feature = features.mean(dim=0)
                mean_feature /= mean_feature.norm()
            label_features.append(mean_feature)
        return torch.stack(label_features)

    def crop_to_subject(self, image: Image.Image, padding_frac: float = 0.04):
        """Removes the background and crops to the bounding box of the
        remaining foreground. Falls back to the original image if
        background removal fails or finds no clear foreground -- this
        should never be the reason a search request errors out.

        Returns (cropped_image, aspect_ratio) -- aspect_ratio is the
        foreground box's width:height, or None if no usable foreground
        was found."""
        try:
            rgba = remove(image.convert("RGB"), session=self._rembg_session)
        except Exception:
            return image, None

        alpha = np.array(rgba.split()[-1])
        ys, xs = np.where(alpha > 16)
        if len(xs) == 0 or len(ys) == 0:
            return image, None

        w, h = image.size
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad_x = int((x1 - x0) * padding_frac)
        pad_y = int((y1 - y0) * padding_frac)

        cropped = image.crop((
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(w, x1 + pad_x),
            min(h, y1 + pad_y),
        ))
        box_w, box_h = (x1 - x0), (y1 - y0)
        aspect_ratio = box_w / box_h if box_h > 0 else None
        return cropped, aspect_ratio

    def analyze_image(self, image: Image.Image, crop: bool = True) -> dict:
        """Single pass over one image: crop to the furniture subject, then
        return both the search embedding and the predicted material/shape
        -- computed from the same cropped image so they stay consistent."""
        image = image.convert("RGB")
        aspect_ratio = None
        if crop:
            image, aspect_ratio = self.crop_to_subject(image)

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # Raw cosine similarities across a handful of prompts are all
            # close together (e.g. 0.24 vs 0.26), so softmax on them alone
            # comes out nearly uniform regardless of the image -- standard
            # CLIP zero-shot scales by the model's learned logit_scale
            # first, which is what actually produces a peaked, meaningful
            # confidence distribution.
            logit_scale = self.model.logit_scale.exp()
            material_sims = logit_scale * (image_features @ self._material_text_features.T).squeeze(0)
            material_probs = material_sims.softmax(dim=-1)
            shape_sims = logit_scale * (image_features @ self._shape_text_features.T).squeeze(0)
            shape_probs = shape_sims.softmax(dim=-1)

        best_material_idx = int(material_probs.argmax())
        best_shape_idx = int(shape_probs.argmax())
        shape_confidence = float(shape_probs[best_shape_idx])

        # Veto an impossible round/square call: those shapes should
        # produce a roughly 1:1 foreground bounding box, so a strongly
        # elongated one means CLIP's top pick is wrong. Re-pick the
        # better-scoring of just rectangular/oval instead of trusting it.
        if aspect_ratio is not None and SHAPE_LABELS[best_shape_idx] in ("round", "square"):
            elongated = aspect_ratio > ELONGATED_ASPECT_RATIO or aspect_ratio < 1 / ELONGATED_ASPECT_RATIO
            if elongated:
                candidates = [i for i, lbl in enumerate(SHAPE_LABELS) if lbl in ("rectangular", "oval")]
                best_shape_idx = max(candidates, key=lambda i: shape_probs[i])
                shape_confidence = float(shape_probs[best_shape_idx])

        return {
            "vector": image_features.squeeze(0).cpu().numpy().tolist(),
            "material": MATERIAL_LABELS[best_material_idx],
            "material_confidence": float(material_probs[best_material_idx]),
            "shape": SHAPE_LABELS[best_shape_idx],
            "shape_confidence": shape_confidence,
        }

    def analyze_image_from_url(self, url: str) -> dict:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content))
        return self.analyze_image(image)

    def analyze_image_from_bytes(self, data: bytes) -> dict:
        image = Image.open(BytesIO(data))
        return self.analyze_image(image)