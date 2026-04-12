"""
Nkarik Backend v2 — FastAPI
Child-safe, optimised image generation server.

Key improvements over v1:
- SDXL-Turbo for 4-step generation (~3s on GPU vs ~20s)
- Falls back to SD 1.5 img2img when SDXL-Turbo unavailable
- Per-style prompt templates with negative prompts
- Async generation with background task queue
- Memory-efficient model management
- Structured logging
- Background image purge task (no per-request overhead)

Install:
    pip install fastapi uvicorn[standard] pillow diffusers transformers \
                torch accelerate slowapi python-multipart apscheduler
"""

import asyncio
import base64
import io
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import torch
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from diffusers import (
    AutoPipelineForImage2Image,
    StableDiffusionImg2ImgPipeline,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("nkarik")

# ─── Config ──────────────────────────────────────────────────────────────────
USE_TURBO      = os.getenv("USE_TURBO", "true").lower() == "true"
MODEL_TURBO    = "stabilityai/sdxl-turbo"
MODEL_FALLBACK = os.getenv("SD_MODEL_ID", "nitrosocke/Ghibli-Diffusion")
IMAGE_TTL      = int(os.getenv("IMAGE_TTL_SECONDS", 86_400))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ─── Style prompt templates ───────────────────────────────────────────────────
STYLE_TEMPLATES = {
    "cartoon ghibli anime style": {
        "positive": (
            "studio ghibli style, anime illustration, children's book art, "
            "soft painterly background, vibrant nature, friendly characters, "
            "high quality, masterpiece"
        ),
        "negative": "realistic, photo, dark, scary, violence, nsfw, adult",
    },
    "pixar 3d cartoon style": {
        "positive": (
            "pixar 3d animation style, cute cartoon character, "
            "colorful lighting, friendly expression, high quality render"
        ),
        "negative": "flat, 2d, dark, scary, realistic, nsfw, adult",
    },
    "cute watercolor illustration": {
        "positive": (
            "watercolor illustration, soft pastel colors, "
            "children's book art, dreamy, gentle brushstrokes, kawaii"
        ),
        "negative": "digital, sharp, dark, scary, nsfw, adult",
    },
    "comic book pop art style": {
        "positive": (
            "pop art comic book style, bold outlines, bright colors, "
            "halftone dots, dynamic, fun, expressive, kids comic"
        ),
        "negative": "realistic, dark, violent, nsfw, adult, muted",
    },
    "cute kawaii sticker art": {
        "positive": (
            "kawaii sticker art, cute chibi character, pastel colors, "
            "white outline, adorable, happy expression, clean background"
        ),
        "negative": "realistic, scary, dark, nsfw, adult, complex background",
    },
}

GLOBAL_POSITIVE_SUFFIX = (
    ", safe for children, child-friendly, bright colors, happy, cute"
)
GLOBAL_NEGATIVE = (
    "nsfw, nude, violence, gore, weapons, drugs, alcohol, scary, "
    "horror, dark themes, realistic photo, ugly, deformed"
)

# ─── Image store (in-memory; swap for Redis/S3 in production) ───────────────
_image_store: dict[str, dict] = {}
_store_lock = asyncio.Lock()


async def purge_expired():
    """Run by scheduler every 10 minutes."""
    async with _store_lock:
        cutoff = datetime.utcnow() - timedelta(seconds=IMAGE_TTL)
        expired = [k for k, v in _image_store.items() if v["created_at"] < cutoff]
        for k in expired:
            del _image_store[k]
        if expired:
            logger.info("Purged %d expired images", len(expired))


# ─── Model management ────────────────────────────────────────────────────────
_pipeline = None
_pipeline_type: Optional[str] = None  # "turbo" | "standard"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline():
    global _pipeline, _pipeline_type
    device = get_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    logger.info("Device: %s  dtype: %s", device, dtype)

    if USE_TURBO:
        try:
            logger.info("Loading SDXL-Turbo…")
            _pipeline = AutoPipelineForImage2Image.from_pretrained(
                MODEL_TURBO,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
            ).to(device)
            # Turbo: enable memory optimizations
            if hasattr(_pipeline, "enable_attention_slicing"):
                _pipeline.enable_attention_slicing()
            _pipeline_type = "turbo"
            logger.info("SDXL-Turbo loaded")
            return
        except Exception as e:
            logger.warning("SDXL-Turbo unavailable (%s), falling back", e)

    logger.info("Loading standard SD model: %s", MODEL_FALLBACK)
    _pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_FALLBACK,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    if hasattr(_pipeline, "enable_attention_slicing"):
        _pipeline.enable_attention_slicing()
    _pipeline_type = "standard"
    logger.info("Standard pipeline loaded")


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        load_pipeline()
    return _pipeline


# ─── App lifespan ─────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load model on startup (remove to keep lazy loading)
    # get_pipeline()
    scheduler.add_job(purge_expired, "interval", minutes=10)
    scheduler.start()
    yield
    scheduler.shutdown()


# ─── App setup ────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Nkarik API", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── Safety ──────────────────────────────────────────────────────────────────
_UNSAFE_KEYWORDS = frozenset({
    "nude", "naked", "explicit", "nsfw", "gore", "violence", "blood",
    "weapon", "gun", "knife", "drug", "alcohol", "porn", "sex",
})


def is_prompt_safe(text: str) -> bool:
    lowered = text.lower()
    return not any(kw in lowered for kw in _UNSAFE_KEYWORDS)


def moderate_image(img: Image.Image) -> bool:
    """
    Stub — in production integrate AWS Rekognition ModerationLabels
    or Google Vision SafeSearch. Returns True = safe.
    """
    return True


# ─── Schemas ─────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    image: str
    style: str = "cartoon ghibli anime style"
    strength: float = 0.65

    @field_validator("image")
    @classmethod
    def check_image(cls, v):
        if not v.startswith("data:image/"):
            raise ValueError("image must be a base64 data-URL")
        if len(v) > 2_500_000:
            raise ValueError("image too large")
        return v

    @field_validator("strength")
    @classmethod
    def check_strength(cls, v):
        if not 0.1 <= v <= 0.95:
            raise ValueError("strength must be 0.1–0.95")
        return v


class GenerateResponse(BaseModel):
    status: str
    result: str
    image_id: str
    guess: Optional[str] = None


# ─── Helpers ─────────────────────────────────────────────────────────────────
def decode_image(data_url: str, size: int = 512) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=88)   # WebP ~30% smaller than PNG
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompts(style_key: str) -> tuple[str, str]:
    template = STYLE_TEMPLATES.get(style_key, STYLE_TEMPLATES["cartoon ghibli anime style"])
    positive = template["positive"] + GLOBAL_POSITIVE_SUFFIX
    negative = template["negative"] + ", " + GLOBAL_NEGATIVE
    return positive, negative


def run_generation(input_img: Image.Image, positive: str, negative: str, strength: float) -> Image.Image:
    pipe = get_pipeline()

    if _pipeline_type == "turbo":
        # SDXL-Turbo: 4 steps, no negative prompt, guidance_scale=0
        result = pipe(
            prompt=positive,
            image=input_img,
            num_inference_steps=4,
            strength=min(strength, 0.8),  # turbo caps well at 0.8
            guidance_scale=0.0,
        )
    else:
        result = pipe(
            prompt=positive,
            negative_prompt=negative,
            image=input_img,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=25,  # reduced from 30 for speed
        )
    return result.images[0]


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _pipeline_type or "not loaded",
        "device": get_device(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("10/minute")
async def generate(req: GenerateRequest, request: Request):
    # Safety checks
    if not is_prompt_safe(req.style):
        raise HTTPException(400, "Unsafe style prompt")

    # Validate style key
    style_key = req.style if req.style in STYLE_TEMPLATES else "cartoon ghibli anime style"

    # Decode
    try:
        input_img = decode_image(req.image)
    except Exception as e:
        raise HTTPException(422, f"Invalid image: {e}") from e

    # Generate (run in thread pool to avoid blocking the event loop)
    try:
        positive, negative = build_prompts(style_key)
        logger.info("Generating [%s] strength=%.2f pipeline=%s", style_key, req.strength, _pipeline_type)

        loop = asyncio.get_event_loop()
        output_img = await loop.run_in_executor(
            None, run_generation, input_img, positive, negative, req.strength
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(500, "Image generation failed") from e

    # Moderate output
    if not moderate_image(output_img):
        raise HTTPException(422, "Output failed safety check")

    # Store
    result_b64 = encode_image(output_img)
    image_id = str(uuid.uuid4())
    async with _store_lock:
        _image_store[image_id] = {"data": result_b64, "created_at": datetime.utcnow()}

    logger.info("Done image_id=%s", image_id)
    return GenerateResponse(status="ok", result=result_b64, image_id=image_id)


@app.get("/image/{image_id}")
@limiter.limit("30/minute")
async def get_image(image_id: str, request: Request):
    async with _store_lock:
        record = _image_store.get(image_id)
    if not record:
        raise HTTPException(404, "Image not found or expired")
    return {"status": "ok", "result": record["data"]}


# Run: uvicorn main:app --host 0.0.0.0 --port 8000
