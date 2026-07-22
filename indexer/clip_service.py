import os
import json
import base64
import requests
from io import BytesIO
from PIL import Image
from openai import AsyncOpenAI
import asyncio

HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY") # Optional: replace local rembg

# Hugging Face Serverless Inference API for CLIP
HF_MODEL_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"

class ServerlessClipService:
    """A lightweight service that offloads ML tasks to cloud APIs instead of local GPU/CPU."""
    
    _instance = None

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    @classmethod
    def get(cls) -> "ServerlessClipService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _get_hf_embedding(self, image_bytes: bytes) -> list[float]:
        """Fetches the 768-dimensional vector from Hugging Face."""
        # Note: run_in_threadpool or async requests should be used in production
        # to prevent blocking the FastAPI event loop.
        response = requests.post(HF_MODEL_URL, headers=self.hf_headers, data=image_bytes)
        response.raise_for_status()
        
        # HF CLIP returns a 1D array of floats
        return response.json()

    async def _get_openai_classification(self, base64_image: str) -> dict:
        """Uses OpenAI Vision to classify material and shape with high accuracy."""
        prompt = """
        Analyze this table and return a strict JSON object with two keys: "material" and "shape".
        Choose "material" from: [marble, travertine stone, granite stone, solid wood, metal, glass, rattan or wicker, upholstered fabric].
        Choose "shape" from: [rectangular, square, round, oval].
        If you are unsure or it is not a table, return "unknown" for both.
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=50
        )
        
        return json.loads(response.choices[0].message.content)

    async def analyze_image_from_bytes(self, data: bytes) -> dict:
        """Main entry point: gets embedding and classification in parallel."""
        # Convert bytes to base64 for OpenAI
        base64_image = base64.b64encode(data).decode('utf-8')
        
        # Run both external API calls concurrently to save time
        embedding_task = self._get_hf_embedding(data)
        classification_task = self._get_openai_classification(base64_image)
        
        vector, classification = await asyncio.gather(embedding_task, classification_task)

        return {
            "vector": vector,
            "material": classification.get("material", "unknown"),
            "shape": classification.get("shape", "unknown"),
            "material_confidence": 0.99 if classification.get("material") != "unknown" else 0.0,
            "shape_confidence": 0.99 if classification.get("shape") != "unknown" else 0.0,
        }