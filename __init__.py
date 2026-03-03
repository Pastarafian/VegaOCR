"""
VegaOCR — The Ultimate Multi-Engine ML-Powered OCR Toolkit

Import directly:
    from vegaocr import VegaOCR

    ocr = VegaOCR()
    text = ocr.read_text("image.png")
"""

from ocr_engine import VegaOCR, OCRResult, Detection

__version__ = "1.0.0"
__all__ = ["VegaOCR", "OCRResult", "Detection"]
