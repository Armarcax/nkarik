# Nkarik v2 🎨

> Draw anything — AI makes it magical. A child-focused creative app.

## What's new in v2

| Area | v1 | v2 |
|---|---|---|
| **AI model** | SD img2img (30 steps, ~20s) | SDXL-Turbo (4 steps, ~3s on GPU) |
| **Image format** | PNG (large) | WebP 88% quality (~30% smaller) |
| **Prompts** | Generic | Per-style templates with negative prompts |
| **Event loop** | Blocking generation | `run_in_executor` (non-blocking async) |
| **Image purge** | Per-request | Scheduled every 10 min (APScheduler) |
| **Canvas tools** | Pen only | Pen + eraser + undo (20 steps) |
| **Loading UX** | Spinner | Animated magic wand + rotating messages |
| **Result UX** | Plain image | Reveal animation + floating particles |
| **Style prompts** | Appended to style string | Structured positive + negative templates |

---

## Quick start

### Backend
```bash
pip install fastapi uvicorn[standard] pillow diffusers transformers \
            torch accelerate slowapi python-multipart apscheduler
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
npm install
npm run dev
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://127.0.0.1:8000` | Backend URL |
| `USE_TURBO` | `true` | Use SDXL-Turbo (faster). Set `false` for SD 1.5 |
| `SD_MODEL_ID` | `nitrosocke/Ghibli-Diffusion` | Fallback SD model |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `IMAGE_TTL_SECONDS` | `86400` | Auto-delete time for stored images |

---

## GitHub workflow

### Branch strategy
```
main          ← production-ready, protected
dev           ← integration branch
feat/xxx      ← feature branches (e.g. feat/undo, feat/guess-mode)
fix/xxx       ← bug fixes
```

### Recommended PR flow
```
feat/my-feature → dev → (review + tests) → main
```

### Commit format
```
feat: add eraser tool
fix: canvas cursor on mobile
perf: switch to SDXL-Turbo
chore: update dependencies
```

---

## Deployment

### Backend — Fly.io (GPU)
```bash
fly launch --name nkarik-api
fly scale vm a100-40gb   # or l40s for cost efficiency
fly deploy
```

### Frontend — Vercel
```bash
vercel --prod
# Set VITE_API_URL env var in Vercel dashboard
```

---

## Phase 2 roadmap

- **AI drawing guesser** — CLIP classifier identifies subject, enriches prompt
- **Style memory** — Per-session drawing fingerprint, no PII stored
- **Gamification** — Badges, levels, unlockable styles
- **Parent dashboard** — History view, style lock, safe-mode toggle
- **Offline PWA** — Service worker, canvas saves to IndexedDB
