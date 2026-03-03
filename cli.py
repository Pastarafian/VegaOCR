"""
VegaOCR — Command-Line Interface

Usage:
    # OCR an image (ensemble mode, best accuracy)
    python cli.py image.png

    # Specify engine
    python cli.py image.png --engine paddle

    # OCR a PDF
    python cli.py document.pdf --engine easy --lang ch

    # Output as JSON
    python cli.py image.png --json

    # Output just the text
    python cli.py image.png --text-only

    # Save results to file
    python cli.py image.png --output results.txt

    # Process multiple files
    python cli.py image1.png image2.jpg scan.pdf
"""

import sys
import json
import argparse
from pathlib import Path

from loguru import logger

from ocr_engine import VegaOCR


def main():
    parser = argparse.ArgumentParser(
        prog="vegaocr",
        description="VegaOCR — Multi-Engine ML-Powered OCR Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py photo.png                    # Ensemble OCR (best accuracy)
  python cli.py scan.pdf --engine paddle     # PaddleOCR on PDF
  python cli.py invoice.jpg --json           # JSON output
  python cli.py *.png --output results.txt   # Batch process, save to file
        """,
    )
    parser.add_argument(
        "files", nargs="+", help="Image or PDF file(s) to process"
    )
    parser.add_argument(
        "--engine", "-e",
        choices=["ensemble", "paddle", "easy", "tesseract"],
        default="ensemble",
        help="OCR engine to use (default: ensemble)",
    )
    parser.add_argument(
        "--lang", "-l",
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--text-only", "-t",
        action="store_true",
        help="Output only extracted text (no metadata)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save output to file",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable image preprocessing",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress log messages",
    )

    args = parser.parse_args()

    if args.quiet:
        logger.remove()

    # Initialize engine
    ocr = VegaOCR(
        engine=args.engine,
        lang=args.lang,
        preprocess_enabled=not args.no_preprocess,
    )

    all_output = []

    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            logger.error(f"File not found: {filepath}")
            continue

        logger.info(f"Processing: {filepath}")

        try:
            if path.suffix.lower() == ".pdf":
                result = ocr.read_pdf(str(path))
            else:
                result = ocr.read(str(path))

            if args.json_output:
                output = json.dumps({
                    "file": str(path),
                    **result.to_dict(),
                }, indent=2, ensure_ascii=False)
            elif args.text_only:
                output = result.full_text
            else:
                # Pretty formatted output
                lines = [
                    f"{'═' * 60}",
                    f"  📄 {path.name}",
                    f"  Engine: {result.engine_used} | "
                    f"Detections: {len(result.detections)} | "
                    f"Time: {result.processing_time_ms:.0f}ms",
                    f"{'─' * 60}",
                ]
                if result.detections:
                    for det in result.detections:
                        conf_bar = "█" * int(det.confidence * 10) + "░" * (10 - int(det.confidence * 10))
                        lines.append(
                            f"  [{conf_bar}] {det.confidence:.0%}  {det.text}"
                        )
                else:
                    lines.append("  (No text detected)")
                lines.append(f"{'═' * 60}")
                lines.append("")
                lines.append(result.full_text)
                output = "\n".join(lines)

            all_output.append(output)
            print(output)

        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")
            if not args.quiet:
                import traceback
                traceback.print_exc()

    # Save to file if requested
    if args.output and all_output:
        output_path = Path(args.output)
        output_path.write_text("\n\n".join(all_output), encoding="utf-8")
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
