<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PaddleOCR-Deep%20Learning-2B6CB0?style=for-the-badge" alt="PaddleOCR">
  <img src="https://img.shields.io/badge/EasyOCR-Multilingual-DC382D?style=for-the-badge" alt="EasyOCR">
  <img src="https://img.shields.io/badge/Tesseract-Google-4285F4?style=for-the-badge" alt="Tesseract">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🔮 VegaOCR</h1>
<p align="center"><b>The ultimate multi-engine, ML-powered OCR toolkit</b></p>
<p align="center">
  A Python library, CLI tool, and stunning web UI that combines the world's best OCR engines<br>
  with intelligent ensemble voting for maximum accuracy.
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Ensemble Mode** | Combines PaddleOCR + EasyOCR + Tesseract with IoU-based weighted voting for best-in-class accuracy |
| ⚡ **PaddleOCR** | Baidu's top-tier deep learning OCR engine (CNN + LSTM), highest accuracy on complex layouts |
| 🌐 **EasyOCR** | Strong multilingual support (80+ languages) and handwriting recognition |
| 📄 **Tesseract** | Google's fast LSTM-based engine, excellent for clean printed documents |
| 🎨 **Web UI** | Beautiful drag-and-drop interface with bounding box overlay and real-time results |
| 🐍 **Python Library** | Import `VegaOCR` directly into any Python program — 3 lines to OCR anything |
| 💻 **CLI Tool** | Batch process files with JSON/text output modes |
| 🔧 **ML Preprocessing** | 8-stage adaptive pipeline: EXIF correction, deskewing, denoising, CLAHE, sharpening, binarization |
| 📑 **PDF Support** | OCR entire PDF documents page-by-page at configurable DPI |
| 🎯 **Confidence Scoring** | Every detection includes confidence scores with color-coded visualization |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Pastarafian/VegaOCR.git
cd VegaOCR

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Tesseract binary must be installed separately for the Tesseract engine.
> Download from: https://github.com/UB-Mannheim/tesseract/wiki  
> PaddleOCR and EasyOCR work out of the box — no extra binaries needed.

### 🎨 Web UI (Drag & Drop)

```bash
python run.py
```

Opens automatically at `http://127.0.0.1:8100` — drag & drop images for instant OCR with bounding box visualization.

### 🐍 Python Library

```python
from ocr_engine import VegaOCR

# Ensemble mode (combines all engines for best accuracy)
ocr = VegaOCR()
result = ocr.read("image.png")
print(result.full_text)

# Use a specific engine
ocr = VegaOCR(engine="paddle")
text = ocr.read_text("photo.jpg")

# OCR a PDF document
result = ocr.read_pdf("document.pdf")
for det in result.detections:
    print(f"[Page {det.page}] {det.text} ({det.confidence:.0%})")

# From a numpy array (OpenCV)
import cv2
img = cv2.imread("screenshot.png")
result = ocr.read(img)

# From raw bytes (e.g., web upload)
with open("scan.png", "rb") as f:
    result = ocr.read_bytes(f.read())
```

### 💻 CLI Tool

```bash
# Basic OCR (ensemble mode)
python cli.py image.png

# Choose engine and language
python cli.py scan.pdf --engine paddle --lang ch

# JSON output
python cli.py invoice.jpg --json

# Plain text only
python cli.py receipt.png --text-only

# Batch process with saved results
python cli.py *.png --output results.txt

# Skip preprocessing (for already-clean images)
python cli.py clean_doc.png --no-preprocess
```

## 🏗️ Architecture

```
VegaOCR/
├── ocr_engine.py        # 🧠 Core library — VegaOCR class (import this)
├── preprocessing.py     # 🔧 8-stage ML image preprocessing pipeline
├── config.py            # ⚙️  Configuration & constants
├── server.py            # 🌐 FastAPI REST API server
├── cli.py               # 💻 Command-line interface
├── run.py               # 🚀 Launch script (server + auto-open browser)
├── requirements.txt     # 📦 Python dependencies
├── static/
│   └── index.html       # 🎨 Web UI (drag-and-drop, bbox overlay)
├── .gitignore
├── LICENSE
└── README.md
```

## 🔧 Preprocessing Pipeline

VegaOCR applies an intelligent 8-stage preprocessing pipeline to maximize OCR accuracy:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | **EXIF Correction** | Auto-rotate based on camera orientation metadata |
| 2 | **Resolution Normalization** | Downscale oversized images with Lanczos interpolation |
| 3 | **Hough Deskewing** | Detect and correct text line skew via Hough Line Transform |
| 4 | **Non-Local Means Denoising** | Remove noise while preserving text edges |
| 5 | **CLAHE Contrast** | Adaptive histogram equalization on luminance channel |
| 6 | **Unsharp Mask Sharpening** | Enhance text edge definition |
| 7 | **Sauvola Binarization** | Adaptive thresholding for text isolation (Tesseract) |
| 8 | **Morphological Cleanup** | Remove speckle noise, fill stroke gaps |

## ⚖️ Engine Comparison

| Engine | Accuracy | Speed | Languages | Best For |
|--------|----------|-------|-----------|----------|
| **Ensemble** | ⭐⭐⭐⭐⭐ | Slower | All | Maximum accuracy, production use |
| **PaddleOCR** | ⭐⭐⭐⭐⭐ | Medium | 80+ | Complex layouts, rotated text, real-world photos |
| **EasyOCR** | ⭐⭐⭐⭐ | Medium | 80+ | Multilingual documents, handwriting |
| **Tesseract** | ⭐⭐⭐ | Fast | 100+ | Clean printed documents, speed-critical tasks |

## 🔌 REST API

When running the web server, the following endpoints are available:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/api/ocr` | Full OCR → JSON with detections, bounding boxes, confidence |
| `POST` | `/api/ocr/text` | OCR → plain text only |
| `GET` | `/api/engines` | List available engines and their status |
| `GET` | `/api/health` | Health check |

### Example API Call

```bash
curl -X POST http://127.0.0.1:8100/api/ocr \
  -F "file=@image.png" \
  -F "engine=ensemble" \
  -F "lang=en"
```

## ⚙️ Configuration

Edit `config.py` to customize behavior:

```python
DEFAULT_ENGINE = "ensemble"       # Default OCR engine
CONFIDENCE_THRESHOLD = 0.40      # Minimum confidence to include a detection
MAX_IMAGE_DIMENSION = 4096       # Auto-downscale limit
DESKEW_ENABLED = True            # Automatic skew correction
DENOISE_STRENGTH = 10            # Denoising intensity
CONTRAST_CLIP_LIMIT = 2.0        # CLAHE clip limit
PORT = 8100                      # Web server port
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with 💜 by <a href="https://github.com/Pastarafian">Pastarafian</a>
</p>
