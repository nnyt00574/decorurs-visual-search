"""
Run once at image build time so the CLIP and background-removal model
weights are baked into this layer, instead of every container start
re-downloading several hundred MB before it can do anything.
"""
from clip_service import ClipService

ClipService.get()
print("Model weights cached.")
