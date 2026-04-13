"""
Nkarik Backend v3 — FastAPI
Child-safe image generation + AI guess + badges.

New in v3:
- CLIP-based drawing guesser ("I think you drew a cat! 🐱")
- NSFW moderation (Falconsai model)
- Badge/reward system (session-based, no PII)
- All in one file — HF Spaces compatible

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
from diffusers import AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from transformers import CLIPModel, CLIPProcessor, pipeline as hf_pipeline

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("nkarik")

# ─── Config ──────────────────────────────────────────────────────────────────
USE_TURBO       = os.getenv("USE_TURBO", "true").lower() == "true"
MODEL_TURBO     = "stabilityai/sdxl-turbo"
MODEL_FALLBACK  = os.getenv("SD_MODEL_ID", "nitrosocke/Ghibli-Diffusion")
IMAGE_TTL       = int(os.getenv("IMAGE_TTL_SECONDS", "86400"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ─── Style prompt templates ───────────────────────────────────────────────────
STYLE_TEMPLATES = {
    "cartoon ghibli anime style": {
        "positive": "studio ghibli style, anime illustration, children's book art, soft painterly, vibrant, friendly characters, masterpiece",
        "negative": "realistic, photo, dark, scary, violence, nsfw, adult",
    },
    "pixar 3d cartoon style": {
        "positive": "pixar 3d animation style, cute cartoon, colorful lighting, friendly expression, high quality render",
        "negative": "flat, 2d, dark, scary, realistic, nsfw, adult",
    },
    "cute watercolor illustration": {
        "positive": "watercolor illustration, soft pastel colors, children's book art, dreamy, gentle brushstrokes, kawaii",
        "negative": "digital, sharp, dark, scary, nsfw, adult",
    },
    "comic book pop art style": {
        "positive": "pop art comic book style, bold outlines, bright colors, halftone dots, dynamic, fun, kids comic",
        "negative": "realistic, dark, violent, nsfw, adult, muted",
    },
    "cute kawaii sticker art": {
        "positive": "kawaii sticker art, cute chibi character, pastel colors, white outline, adorable, happy expression",
        "negative": "realistic, scary, dark, nsfw, adult, complex background",
    },
}

GLOBAL_POSITIVE_SUFFIX = ", safe for children, child-friendly, bright colors, happy, cute"
GLOBAL_NEGATIVE        = "nsfw, nude, violence, gore, weapons, drugs, alcohol, scary, horror, ugly, deformed"

# ─── CLIP categories for drawing guesser ─────────────────────────────────────
GUESS_CATEGORIES = [
    "a cat", "a dog", "a house", "a tree", "a car", "a sun", "a flower",
    "a fish", "a bird", "a boat", "a dinosaur", "a robot", "a dragon",
    "a horse", "a butterfly", "a rainbow", "a star", "a heart", "a person",
    "a mountain", "an elephant", "a lion", "a rocket", "a castle",
]

FUN_FACTS = {
    "a cat":       "Cats sleep 13–16 hours a day! 😺",
    "a dog":       "Dogs can recognize over 150 words! 🐶",
    "a dinosaur":  "T-Rex had arms too short to clap! 🦖",
    "a dragon":    "Dragons appear in stories from every culture! 🐉",
    "a butterfly": "Butterflies taste with their feet! 🦋",
    "a rainbow":   "You can never reach the end of a rainbow! 🌈",
    "a rocket":    "Rockets travel at 28,000 km/h! 🚀",
    "a sun":       "The sun is 4.6 billion years old! ☀️",
    "a flower":    "Some flowers only bloom at night! 🌸",
    "a fish":      "Fish have been on Earth for 500 million years! 🐟",
}

DEFAULT_FUN_FACT = "You have an amazing imagination! 🎨"

# ─── Badge definitions ────────────────────────────────────────────────────────
BADGES = {
    "first_drawing": {"emoji": "🎨", "name": "First Steps",   "desc": "You created your first artwork!"},
    "streak_3":      {"emoji": "🔥", "name": "On Fire",       "desc": "Created 3 artworks in a row!"},
    "explorer":      {"emoji": "🌍", "name": "Explorer",      "desc": "Tried all art styles!"},
    "guesser":       {"emoji": "🎯", "name": "Mind Reader",   "desc": "AI guessed your drawing correctly!"},
    "sharer":        {"emoji": "📤", "name": "Sharing Star",  "desc": "Shared your first artwork!"},
    "drawer_10":     {"emoji": "🏆", "name": "Masterpiece",   "desc": "Created 10 artworks!"},
}

# ─── In-memory stores ─────────────────────────────────────────────────────────
_image_store:   dict[str, dict] = {}
_session_store: dict[str, dict] = {}   # badge / stats per session
_store_lock   = asyncio.Lock()


async def purge_expired():
    async with _store_lock:
        cutoff  = datetime.utcnow() - timedelta(seconds=IMAGE_TTL)
        expired = [k for k, v in _image_store.items() if v["created_at"] < cutoff]
        for k in expired:
            del _image_store[k]
        if expired:
            logger.info("Purged %d expired images", len(expired))


# ─── Model management ─────────────────────────────────────────────────────────
_sd_pipeline        = None
_sd_pipeline_type   = None
_clip_model         = None
_clip_processor     = None
_nsfw_classifier    = None


def get_device():
    if torch.cuda.is_available():  return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_sd():
    global _sd_pipeline, _sd_pipeline_type
    device = get_device()
    dtype  = torch.float16 if device in ("cuda", "mps") else torch.float32

    if USE_TURBO:
        try:
            logger.info("Loading SDXL-Turbo…")
            _sd_pipeline = AutoPipelineForImage2Image.from_pretrained(
                MODEL_TURBO, torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
            ).to(device)
            if hasattr(_sd_pipeline, "enable_attention_slicing"):
                _sd_pipeline.enable_attention_slicing()
            _sd_pipeline_type = "turbo"
            logger.info("SDXL-Turbo loaded on %s", device)
            return
        except Exception as e:
            logger.warning("SDXL-Turbo failed (%s), falling back", e)

    logger.info("Loading SD fallback: %s", MODEL_FALLBACK)
    _sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_FALLBACK, torch_dtype=dtype, safety_checker=None,
    ).to(device)
    if hasattr(_sd_pipeline, "enable_attention_slicing"):
        _sd_pipeline.enable_attention_slicing()
    _sd_pipeline_type = "standard"
    logger.info("SD loaded on %s", device)


def get_sd():
    global _sd_pipeline
    if _sd_pipeline is None:
        load_sd()
    return _sd_pipeline


def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        logger.info("Loading CLIP…")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP loaded")
    return _clip_model, _clip_processor


def get_nsfw():
    global _nsfw_classifier
    if _nsfw_classifier is None:
        logger.info("Loading NSFW classifier…")
        _nsfw_classifier = hf_pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=-1,
        )
        logger.info("NSFW classifier loaded")
    return _nsfw_classifier


# ─── App setup ────────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(purge_expired, "interval", minutes=10)
    scheduler.start()
    yield
    scheduler.shutdown()


limiter = Limiter(key_func=get_remote_address)
app     = FastAPI(title="Nkarik API", version="3.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── Safety helpers ───────────────────────────────────────────────────────────
_UNSAFE_KEYWORDS = frozenset({
    "nude", "naked", "explicit", "nsfw", "gore", "violence", "blood",
    "weapon", "gun", "knife", "drug", "alcohol", "porn", "sex",
})


def is_text_safe(text: str) -> bool:
    return not any(kw in text.lower() for kw in _UNSAFE_KEYWORDS)


async def is_image_safe(img: Image.Image) -> bool:
    """Run NSFW classifier — returns True if safe."""
    try:
        loop       = asyncio.get_event_loop()
        classifier = get_nsfw()
        results    = await loop.run_in_executor(None, classifier, img)
        nsfw_score = next((r["score"] for r in results if r["label"] == "nsfw"), 0.0)
        return nsfw_score < 0.3
    except Exception as e:
        logger.warning("NSFW check failed (%s) — allowing", e)
        return True


# ─── Image helpers ────────────────────────────────────────────────────────────
def decode_image(data_url: str, size: int = 512) -> Image.Image:
    _, encoded = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=88)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompts(style_key: str) -> tuple[str, str]:
    tpl      = STYLE_TEMPLATES.get(style_key, STYLE_TEMPLATES["cartoon ghibli anime style"])
    positive = tpl["positive"] + GLOBAL_POSITIVE_SUFFIX
    negative = tpl["negative"] + ", " + GLOBAL_NEGATIVE
    return positive, negative


def run_sd(input_img, positive, negative, strength) -> Image.Image:
    pipe = get_sd()
    if _sd_pipeline_type == "turbo":
        return pipe(
            prompt=positive, image=input_img,
            num_inference_steps=4,
            strength=min(strength, 0.8),
            guidance_scale=0.0,
        ).images[0]
    return pipe(
        prompt=positive, negative_prompt=negative,
        image=input_img, strength=strength,
        guidance_scale=7.5, num_inference_steps=25,
    ).images[0]


# ─── Session / badge helpers ──────────────────────────────────────────────────
def get_session(session_id: str) -> dict:
    if session_id not in _session_store:
        _session_store[session_id] = {
            "total":   0,
            "styles":  set(),
            "badges":  [],
            "streak":  0,
            "last_at": None,
        }
    return _session_store[session_id]


def check_badges(session: dict, action: str) -> list[dict]:
    """Return list of newly earned badges."""
    earned     = {b["id"] for b in session["badges"]}
    new_badges = []

    def award(badge_id: str):
        if badge_id not in earned:
            b = {**BADGES[badge_id], "id": badge_id, "earned_at": datetime.utcnow().isoformat()}
            session["badges"].append(b)
            new_badges.append(b)

    if action == "generate":
        if session["total"] == 1:                            award("first_drawing")
        if session["total"] >= 10:                           award("drawer_10")
        if session["streak"] >= 3:                           award("streak_3")
        if len(session["styles"]) >= len(STYLE_TEMPLATES):  award("explorer")

    if action == "guess_correct":  award("guesser")
    if action == "share":          award("sharer")

    return new_badges


# ─── Schemas ─────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    image:      str
    style:      str   = "cartoon ghibli anime style"
    strength:   float = 0.65
    session_id: str   = ""

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
    status:     str
    result:     str
    image_id:   str
    guess:      Optional[str]       = None
    fun_fact:   Optional[str]       = None
    new_badges: list                = []
    stats:      dict                = {}


class GuessRequest(BaseModel):
    image:      str
    session_id: str = ""


class GuessResponse(BaseModel):
    guess:       str
    confidence:  float
    alternatives: list[str]
    fun_fact:    str


class ConfirmGuessRequest(BaseModel):
    session_id:  str
    was_correct: bool


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   _sd_pipeline_type or "not loaded",
        "device":  get_device(),
        "time":    datetime.utcnow().isoformat(),
    }


@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("10/minute")
async def generate(req: GenerateRequest, request: Request):
    # Safety checks
    if not is_text_safe(req.style):
        raise HTTPException(400, "Unsafe style prompt")

    style_key = req.style if req.style in STYLE_TEMPLATES else "cartoon ghibli anime style"

    # Decode input
    try:
        input_img = decode_image(req.image)
    except Exception as e:
        raise HTTPException(422, f"Invalid image: {e}") from e

    # Moderate input
    if not await is_image_safe(input_img):
        raise HTTPException(400, "Image failed safety check")

    # Generate
    try:
        positive, negative = build_prompts(style_key)
        logger.info("Generating [%s] strength=%.2f", style_key, req.strength)
        loop       = asyncio.get_event_loop()
        output_img = await loop.run_in_executor(
            None, run_sd, input_img, positive, negative, req.strength
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(500, "Image generation failed") from e

    # Moderate output
    if not await is_image_safe(output_img):
        raise HTTPException(422, "Output failed safety check")

    # CLIP guess (run in parallel with encode)
    guess_text, fun_fact = None, None
    try:
        guess_text, _, fun_fact = await _run_clip_guess(input_img)
    except Exception as e:
        logger.warning("Guess failed: %s", e)

    # Encode and store
    result_b64 = encode_image(output_img)
    image_id   = str(uuid.uuid4())
    async with _store_lock:
        _image_store[image_id] = {"data": result_b64, "created_at": datetime.utcnow()}

    # Update session + badges
    sid     = req.session_id or image_id
    session = get_session(sid)
    session["total"]  += 1
    session["styles"].add(style_key)
    now = datetime.utcnow().date()
    if session["last_at"] == str(now):
        pass
    elif session["last_at"] == str(now - timedelta(days=1)):
        session["streak"] += 1
    else:
        session["streak"] = 1
    session["last_at"] = str(now)
    new_badges = check_badges(session, "generate")

    logger.info("Done image_id=%s session=%s total=%d", image_id, sid, session["total"])

    return GenerateResponse(
        status="ok",
        result=result_b64,
        image_id=image_id,
        guess=guess_text,
        fun_fact=fun_fact,
        new_badges=new_badges,
        stats={
            "total":  session["total"],
            "streak": session["streak"],
            "badges": len(session["badges"]),
        },
    )


@app.post("/guess", response_model=GuessResponse)
@limiter.limit("30/minute")
async def guess_drawing(req: GuessRequest, request: Request):
    """Standalone guess endpoint — call before generating for interactive feedback."""
    try:
        img = decode_image(req.image)
    except Exception as e:
        raise HTTPException(422, f"Invalid image: {e}") from e

    guess, alts, fun_fact = await _run_clip_guess(img)
    return GuessResponse(guess=guess, confidence=0.0, alternatives=alts, fun_fact=fun_fact)


@app.post("/guess/confirm")
async def confirm_guess(req: ConfirmGuessRequest):
    """Child confirms whether AI guessed correctly — awards badge if yes."""
    new_badges = []
    if req.was_correct:
        session    = get_session(req.session_id)
        new_badges = check_badges(session, "guess_correct")
    return {"status": "ok", "new_badges": new_badges}


@app.get("/session/{session_id}")
async def get_session_stats(session_id: str):
    """Return badge + stats for a session."""
    session = get_session(session_id)
    return {
        "total":   session["total"],
        "streak":  session["streak"],
        "badges":  session["badges"],
        "styles":  list(session["styles"]),
    }


@app.get("/image/{image_id}")
@limiter.limit("30/minute")
async def get_image(image_id: str, request: Request):
    async with _store_lock:
        record = _image_store.get(image_id)
    if not record:
        raise HTTPException(404, "Image not found or expired")
    return {"status": "ok", "result": record["data"]}


# ─── CLIP helper ──────────────────────────────────────────────────────────────
async def _run_clip_guess(img: Image.Image) -> tuple[str, list[str], str]:
    loop = asyncio.get_event_loop()

    def _clip_sync():
        model, processor = get_clip()
        inputs  = processor(text=GUESS_CATEGORIES, images=img, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs   = outputs.logits_per_image.softmax(dim=1)
        top3    = probs.topk(3)
        top_cats = [GUESS_CATEGORIES[i] for i in top3.indices[0]]
        return top_cats

    top_cats = await loop.run_in_executor(None, _clip_sync)
    best     = top_cats[0]
    alts     = top_cats[1:]
    fun_fact = FUN_FACTS.get(best, DEFAULT_FUN_FACT)
    return best, alts, fun_fact


# ─── Run: uvicorn main:app --host 0.0.0.0 --port 7860 ────────────────────────
