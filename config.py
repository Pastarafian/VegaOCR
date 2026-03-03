"""
VegaOCR — Configuration & Constants
"""
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Engine Defaults ─────────────────────────────────────────────────────────
DEFAULT_ENGINE = "ensemble"              # "paddle", "easy", "tesseract", "ensemble"
DEFAULT_LANG = "en"
CONFIDENCE_THRESHOLD = 0.40              # minimum confidence to include a detection
ENSEMBLE_AGREEMENT_THRESHOLD = 0.55      # weighted agreement threshold

# ─── Preprocessing ───────────────────────────────────────────────────────────
AUTO_PREPROCESS = True
MAX_IMAGE_DIMENSION = 4096               # downscale if larger than this
DESKEW_ENABLED = True
DENOISE_STRENGTH = 10                    # cv2 fastNlMeansDenoising strength
CONTRAST_CLIP_LIMIT = 2.0               # CLAHE clip limit
CONTRAST_TILE_SIZE = (8, 8)              # CLAHE tile grid size
SHARPEN_AMOUNT = 1.5                     # unsharp mask amount

# ─── Upload Limits ───────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif",
    ".webp", ".gif", ".pdf"
}

# ─── Server ──────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8100
