"""
VegaOCR — FastAPI Web Server

Provides REST API and serves the web UI for drag-and-drop OCR processing.

Endpoints:
    GET  /                   → Web UI
    POST /api/ocr            → Process image/PDF, return JSON results
    POST /api/ocr/text       → Process image/PDF, return plain text only
    GET  /api/engines        → List available engines
    GET  /api/health         → Health check
    GET  /uploads/{filename} → Serve uploaded files (for bbox overlay)
"""

from __future__ import annotations

import uuid
import shutil
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from loguru import logger

import config
from ocr_engine import VegaOCR

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="VegaOCR",
    description="Multi-engine ML-powered OCR utility",
    version="1.0.0",
)

# Serve static files
config.STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# Engine cache (lazy loaded)
_engine_cache: dict[str, VegaOCR] = {}


def _get_engine(engine: str, lang: str) -> VegaOCR:
    """Get or create a cached VegaOCR instance."""
    key = f"{engine}:{lang}"
    if key not in _engine_cache:
        _engine_cache[key] = VegaOCR(engine=engine, lang=lang)
    return _engine_cache[key]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    html_path = config.STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "Web UI not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/engines")
async def list_engines():
    """List available OCR engines and their status."""
    engines = []
    engine_info = [
        ("paddle", "PaddleOCR", "Highest accuracy deep learning engine (Baidu)"),
        ("easy", "EasyOCR", "Strong multilingual & handwriting support"),
        ("tesseract", "Tesseract", "Fast baseline for clean printed documents"),
        ("ensemble", "Ensemble", "Combines all engines with weighted voting (best overall)"),
    ]
    for eid, name, desc in engine_info:
        available = True
        try:
            if eid == "paddle":
                importlib.import_module("paddleocr")
            elif eid == "easy":
                importlib.import_module("easyocr")
            elif eid == "tesseract":
                importlib.import_module("pytesseract")
        except ImportError:
            if eid != "ensemble":
                available = False

        engines.append({
            "id": eid, "name": name,
            "description": desc, "available": available,
        })
    return {"engines": engines}


import importlib


@app.post("/api/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    engine: str = Form("ensemble"),
    lang: str = Form("en"),
):
    """
    Process an uploaded image or PDF through OCR.
    Returns full JSON result with detections, bounding boxes, confidence scores.
    """
    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
        )

    # Validate engine
    valid_engines = {"paddle", "easy", "tesseract", "ensemble"}
    if engine not in valid_engines:
        raise HTTPException(400, f"Invalid engine '{engine}'. Choose: {valid_engines}")

    # Save upload
    file_id = uuid.uuid4().hex[:12]
    save_name = f"{file_id}{ext}"
    save_path = config.UPLOAD_DIR / save_name

    try:
        contents = await file.read()
        if len(contents) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (max {config.MAX_UPLOAD_SIZE_MB}MB)")

        with open(save_path, "wb") as f:
            f.write(contents)

        logger.info(f"Processing: {file.filename} ({len(contents)} bytes) engine={engine}")

        # Run OCR
        ocr = _get_engine(engine, lang)

        if ext == ".pdf":
            result = ocr.read_pdf(str(save_path))
        else:
            result = ocr.read(str(save_path))

        response = result.to_dict()
        response["filename"] = file.filename
        response["file_id"] = file_id
        response["upload_url"] = f"/uploads/{save_name}"

        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR failed: {traceback.format_exc()}")
        raise HTTPException(500, f"OCR processing failed: {str(e)}")


@app.post("/api/ocr/text")
async def run_ocr_text(
    file: UploadFile = File(...),
    engine: str = Form("ensemble"),
    lang: str = Form("en"),
):
    """Process an uploaded image and return plain text only."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'.")

    file_id = uuid.uuid4().hex[:12]
    save_path = config.UPLOAD_DIR / f"{file_id}{ext}"

    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        ocr = _get_engine(engine, lang)
        if ext == ".pdf":
            result = ocr.read_pdf(str(save_path))
        else:
            result = ocr.read(str(save_path))

        return {"text": result.full_text, "engine": engine}

    except Exception as e:
        logger.error(f"OCR text failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    """Serve an uploaded file (for the frontend to display with bbox overlay)."""
    file_path = config.UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting VegaOCR server at http://{config.HOST}:{config.PORT}")
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
    )
