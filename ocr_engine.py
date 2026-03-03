"""
VegaOCR — Multi-Engine OCR Core

This is the primary importable Python utility. Usage:

    from ocr_engine import VegaOCR

    ocr = VegaOCR()                          # ensemble mode (best accuracy)
    ocr = VegaOCR(engine="paddle")           # single engine
    results = ocr.read("image.png")          # returns list of Detection objects
    text = ocr.read_text("image.png")        # returns plain extracted text
    results = ocr.read_pdf("document.pdf")   # per-page OCR

Each Detection contains:
    - text (str)
    - confidence (float, 0–1)
    - bbox (list of 4 [x,y] points)
    - engine (str, which engine detected it)
"""

from __future__ import annotations

import time
import importlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

import config
from preprocessing import (
    load_image,
    preprocess,
    preprocess_for_all_engines,
)


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single text detection from an OCR engine."""
    text: str
    confidence: float
    bbox: list                      # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    engine: str = "unknown"
    page: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OCRResult:
    """Complete result from an OCR run."""
    detections: list[Detection] = field(default_factory=list)
    full_text: str = ""
    engine_used: str = ""
    processing_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    page_count: int = 1
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "full_text": self.full_text,
            "engine_used": self.engine_used,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "page_count": self.page_count,
            "metadata": self.metadata,
        }


# ─── Engine Wrappers ────────────────────────────────────────────────────────

class PaddleEngine:
    """PaddleOCR wrapper — highest accuracy deep learning engine."""

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._reader = None

    def _ensure_loaded(self):
        if self._reader is None:
            logger.info("Loading PaddleOCR engine...")
            from paddleocr import PaddleOCR
            self._reader = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                show_log=False,
                use_gpu=True,          # falls back to CPU if unavailable
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=16,
            )
            logger.info("PaddleOCR loaded")

    def run(self, img: np.ndarray) -> list[Detection]:
        self._ensure_loaded()
        results = self._reader.ocr(img, cls=True)
        detections = []

        if results and results[0]:
            for line in results[0]:
                bbox_raw, (text, conf) = line
                if conf < config.CONFIDENCE_THRESHOLD:
                    continue
                bbox = [[int(p[0]), int(p[1])] for p in bbox_raw]
                detections.append(Detection(
                    text=text.strip(),
                    confidence=round(float(conf), 4),
                    bbox=bbox,
                    engine="paddle",
                ))

        return detections


class EasyEngine:
    """EasyOCR wrapper — strong multilingual & handwriting support."""

    def __init__(self, lang: str = "en"):
        self.lang_list = [lang] if lang != "en" else ["en"]
        self._reader = None

    def _ensure_loaded(self):
        if self._reader is None:
            logger.info("Loading EasyOCR engine...")
            import easyocr
            self._reader = easyocr.Reader(
                self.lang_list,
                gpu=True,              # falls back to CPU
                verbose=False,
            )
            logger.info("EasyOCR loaded")

    def run(self, img: np.ndarray) -> list[Detection]:
        self._ensure_loaded()
        results = self._reader.readtext(img, detail=1)
        detections = []

        for (bbox_raw, text, conf) in results:
            if conf < config.CONFIDENCE_THRESHOLD:
                continue
            # EasyOCR returns [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            bbox = [[int(p[0]), int(p[1])] for p in bbox_raw]
            detections.append(Detection(
                text=text.strip(),
                confidence=round(float(conf), 4),
                bbox=bbox,
                engine="easy",
            ))

        return detections


class TesseractEngine:
    """Tesseract wrapper — fast baseline for clean documents."""

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def run(self, img: np.ndarray) -> list[Detection]:
        import pytesseract

        # If color, convert to grayscale for Tesseract
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        data = pytesseract.image_to_data(
            gray, lang=self.lang, output_type=pytesseract.Output.DICT
        )
        detections = []
        n = len(data["text"])

        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < config.CONFIDENCE_THRESHOLD * 100:
                continue

            x, y, w, h = (
                data["left"][i], data["top"][i],
                data["width"][i], data["height"][i],
            )
            bbox = [
                [x, y], [x + w, y],
                [x + w, y + h], [x, y + h],
            ]
            detections.append(Detection(
                text=text,
                confidence=round(conf / 100.0, 4),
                bbox=bbox,
                engine="tesseract",
            ))

        return detections


# ─── Ensemble Logic ─────────────────────────────────────────────────────────

def _bbox_iou(bbox1: list, bbox2: list) -> float:
    """Compute IoU between two quadrilateral bounding boxes (approximated as rects)."""
    def _to_rect(bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    x1a, y1a, x2a, y2a = _to_rect(bbox1)
    x1b, y1b, x2b, y2b = _to_rect(bbox2)

    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _merge_detections(all_detections: list[list[Detection]],
                      engine_weights: dict[str, float]) -> list[Detection]:
    """
    Merge detections from multiple engines using weighted voting.

    For overlapping detections (IoU > 0.4), the text with highest
    weighted confidence wins, and the confidence is boosted by agreement.
    """
    # Flatten all detections
    flat = []
    for det_list in all_detections:
        flat.extend(det_list)

    if not flat:
        return []

    # Sort by confidence descending
    flat.sort(key=lambda d: d.confidence * engine_weights.get(d.engine, 1.0),
              reverse=True)

    merged = []
    used = [False] * len(flat)

    for i, det_i in enumerate(flat):
        if used[i]:
            continue

        group = [det_i]
        used[i] = True

        for j in range(i + 1, len(flat)):
            if used[j]:
                continue
            if _bbox_iou(det_i.bbox, flat[j].bbox) > 0.4:
                group.append(flat[j])
                used[j] = True

        # Pick the best text by weighted confidence
        best = max(group,
                   key=lambda d: d.confidence * engine_weights.get(d.engine, 1.0))

        # Boost confidence based on agreement
        engines_agreeing = len(set(d.engine for d in group))
        agreement_boost = min(0.15 * (engines_agreeing - 1), 0.30)
        boosted_conf = min(best.confidence + agreement_boost, 1.0)

        merged.append(Detection(
            text=best.text,
            confidence=round(boosted_conf, 4),
            bbox=best.bbox,
            engine=f"ensemble({'+'.join(d.engine for d in group)})",
        ))

    return merged


# ─── Main OCR Class ─────────────────────────────────────────────────────────

ENGINE_WEIGHTS = {
    "paddle": 1.0,
    "easy": 0.85,
    "tesseract": 0.70,
}


class VegaOCR:
    """
    Multi-engine OCR utility.

    Usage:
        ocr = VegaOCR()                           # ensemble (best)
        ocr = VegaOCR(engine="paddle")             # single engine
        ocr = VegaOCR(engine="paddle", lang="ch")  # Chinese

        result = ocr.read("image.png")             # full OCR result
        text = ocr.read_text("image.png")          # just the text
        pages = ocr.read_pdf("doc.pdf")            # PDF OCR
    """

    def __init__(
        self,
        engine: str = config.DEFAULT_ENGINE,
        lang: str = config.DEFAULT_LANG,
        preprocess_enabled: bool = True,
    ):
        self.engine_name = engine
        self.lang = lang
        self.preprocess_enabled = preprocess_enabled
        self._engines: dict = {}
        self._init_engines()

    def _init_engines(self):
        """Lazily initialize requested engine(s)."""
        if self.engine_name in ("paddle", "ensemble"):
            self._engines["paddle"] = PaddleEngine(self.lang)
        if self.engine_name in ("easy", "ensemble"):
            self._engines["easy"] = EasyEngine(self.lang)
        if self.engine_name in ("tesseract", "ensemble"):
            tess_lang = "eng" if self.lang == "en" else self.lang
            self._engines["tesseract"] = TesseractEngine(tess_lang)

    def read(self, source, page: int = 0) -> OCRResult:
        """
        Run OCR on an image file path, numpy array, or PIL Image.

        Args:
            source: str/Path (file path), np.ndarray (BGR image), or PIL.Image
            page: page number (for multi-page tracking)

        Returns:
            OCRResult with all detections, full text, timing info
        """
        t0 = time.perf_counter()

        # Load image
        if isinstance(source, (str, Path)):
            img = load_image(str(source))
        elif isinstance(source, np.ndarray):
            img = source.copy()
        else:
            # Assume PIL Image
            img = np.array(source)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # Preprocess
        if self.preprocess_enabled and self.engine_name == "ensemble":
            preprocessed = preprocess_for_all_engines(img)
        elif self.preprocess_enabled:
            preprocessed = {self.engine_name: preprocess(img, self.engine_name)}
        else:
            preprocessed = {name: img for name in self._engines}

        # Run engine(s)
        if self.engine_name == "ensemble":
            all_detections = []
            for name, engine in self._engines.items():
                try:
                    engine_img = preprocessed.get(name, img)
                    dets = engine.run(engine_img)
                    logger.info(f"  {name}: {len(dets)} detections")
                    all_detections.append(dets)
                except Exception as e:
                    logger.warning(f"  {name} engine failed: {e}")
                    all_detections.append([])

            detections = _merge_detections(all_detections, ENGINE_WEIGHTS)
        else:
            engine = self._engines[self.engine_name]
            engine_img = preprocessed.get(self.engine_name, img)
            detections = engine.run(engine_img)

        # Sort detections top-to-bottom, left-to-right
        detections.sort(key=lambda d: (
            min(p[1] for p in d.bbox),
            min(p[0] for p in d.bbox),
        ))

        # Set page number
        for d in detections:
            d.page = page

        # Build full text
        full_text = "\n".join(d.text for d in detections)

        elapsed = (time.perf_counter() - t0) * 1000

        return OCRResult(
            detections=detections,
            full_text=full_text,
            engine_used=self.engine_name,
            processing_time_ms=elapsed,
            image_width=w,
            image_height=h,
            page_count=1,
            metadata={"lang": self.lang, "preprocessed": self.preprocess_enabled},
        )

    def read_text(self, source) -> str:
        """Convenience: return just the extracted text."""
        return self.read(source).full_text

    def read_pdf(self, pdf_path: str, dpi: int = 300) -> OCRResult:
        """
        OCR every page of a PDF. Returns a combined OCRResult.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering PDF pages

        Returns:
            OCRResult with all detections across all pages
        """
        import fitz  # PyMuPDF

        t0 = time.perf_counter()
        doc = fitz.open(pdf_path)
        all_detections = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to numpy array
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            page_result = self.read(img, page=page_num)
            all_detections.extend(page_result.detections)
            logger.info(
                f"PDF page {page_num + 1}/{len(doc)}: "
                f"{len(page_result.detections)} detections"
            )

        doc.close()

        full_text = "\n".join(d.text for d in all_detections)
        elapsed = (time.perf_counter() - t0) * 1000

        return OCRResult(
            detections=all_detections,
            full_text=full_text,
            engine_used=self.engine_name,
            processing_time_ms=elapsed,
            page_count=len(doc) if doc else 0,
            metadata={"lang": self.lang, "source": "pdf", "dpi": dpi},
        )

    def read_bytes(self, image_bytes: bytes, filename: str = "") -> OCRResult:
        """OCR from raw bytes (e.g., uploaded file)."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes")
        return self.read(img)
