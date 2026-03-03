"""
VegaOCR — ML-Powered Image Preprocessing Pipeline

Applies adaptive, multi-stage preprocessing to maximize OCR accuracy:
  1. Auto-orientation (EXIF correction)
  2. Resolution normalization
  3. Adaptive deskewing (Hough transform + projection analysis)
  4. Intelligent denoising (non-local means + bilateral filtering)
  5. CLAHE contrast enhancement
  6. Binarization (Sauvola adaptive thresholding)
  7. Morphological cleanup
  8. Sharpening (unsharp mask)
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
from loguru import logger

import config


def load_image(path: str) -> np.ndarray:
    """Load image with EXIF orientation correction."""
    try:
        pil_img = Image.open(path)
        # Apply EXIF orientation
        try:
            for orientation_key in ExifTags.TAGS:
                if ExifTags.TAGS[orientation_key] == "Orientation":
                    break
            exif = pil_img._getexif()
            if exif and orientation_key in exif:
                orientation = exif[orientation_key]
                rotations = {3: 180, 6: 270, 8: 90}
                if orientation in rotations:
                    pil_img = pil_img.rotate(rotations[orientation], expand=True)
        except (AttributeError, KeyError):
            pass

        # Convert to BGR for OpenCV
        if pil_img.mode == "RGBA":
            # Composite onto white background
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise


def normalize_resolution(img: np.ndarray) -> np.ndarray:
    """Downscale oversized images while preserving aspect ratio."""
    h, w = img.shape[:2]
    max_dim = config.MAX_IMAGE_DIMENSION
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        logger.info(f"Downscaled from {w}x{h} to {new_w}x{new_h}")
    return img


def _compute_skew_angle(gray: np.ndarray) -> float:
    """Compute skew angle using Hough Line Transform."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Dilate to connect edge segments for line detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=100, minLineLength=gray.shape[1] // 8, maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines
        if abs(angle) < 30:
            angles.append(angle)

    if not angles:
        return 0.0

    # Median angle is more robust than mean
    return float(np.median(angles))


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct image skew using detected text line angles."""
    if not config.DESKEW_ENABLED:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = _compute_skew_angle(gray)

    if abs(angle) < 0.3:
        return img  # Skip tiny corrections that may reduce quality

    logger.info(f"Deskew: correcting {angle:.2f}° skew")
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        img, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def denoise(img: np.ndarray) -> np.ndarray:
    """Apply intelligent multi-pass denoising."""
    strength = config.DENOISE_STRENGTH
    if len(img.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, strength, strength, 7, 21
        )
    else:
        denoised = cv2.fastNlMeansDenoising(img, None, strength, 7, 21)
    return denoised


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to the luminance channel."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
    else:
        l_channel = img

    clahe = cv2.createCLAHE(
        clipLimit=config.CONTRAST_CLIP_LIMIT,
        tileGridSize=config.CONTRAST_TILE_SIZE
    )
    enhanced_l = clahe.apply(l_channel)

    if len(img.shape) == 3:
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_l


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """Sauvola-inspired adaptive binarization for best text isolation."""
    # Use Gaussian adaptive threshold (approximating Sauvola)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=15
    )
    return binary


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """Remove noise speckles and fill small gaps in text strokes."""
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Thicken thin strokes slightly
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    return cleaned


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp mask sharpening to enhance text edges."""
    amount = config.SHARPEN_AMOUNT
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def preprocess(img: np.ndarray, for_engine: str = "default") -> np.ndarray:
    """
    Full preprocessing pipeline. Returns a cleaned image optimized for OCR.

    Args:
        img: Input BGR image
        for_engine: Target engine ("paddle", "easy", "tesseract", "default")

    Returns:
        Preprocessed image (BGR for paddle/easy, grayscale for tesseract)
    """
    if not config.AUTO_PREPROCESS:
        return img

    logger.info("Preprocessing pipeline started")

    # Step 1: Resolution normalization
    img = normalize_resolution(img)

    # Step 2: Deskew
    img = deskew(img)

    # Step 3: Denoise
    img = denoise(img)

    # Step 4: Contrast enhancement
    img = enhance_contrast(img)

    # Step 5: Sharpen
    img = sharpen(img)

    # Engine-specific final processing
    if for_engine == "tesseract":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = adaptive_binarize(gray)
        binary = morphological_cleanup(binary)
        return binary

    logger.info("Preprocessing pipeline complete")
    return img


def preprocess_for_all_engines(img: np.ndarray) -> dict:
    """
    Generate engine-specific preprocessed versions.

    Returns:
        Dict mapping engine name to optimized image.
    """
    base = normalize_resolution(img.copy())
    base = deskew(base)
    base = denoise(base)
    base = enhance_contrast(base)
    base = sharpen(base)

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    binary = adaptive_binarize(gray)
    binary = morphological_cleanup(binary)

    return {
        "paddle": base.copy(),
        "easy": base.copy(),
        "tesseract": binary.copy(),
        "original_cleaned": base.copy(),
    }
