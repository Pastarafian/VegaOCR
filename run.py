"""
VegaOCR — Launch Script
Start the web server and open the browser.
"""

import webbrowser
import threading
import time

import uvicorn
from loguru import logger

import config


def open_browser():
    """Open browser after a short delay to let the server start."""
    time.sleep(1.5)
    url = f"http://{config.HOST}:{config.PORT}"
    logger.info(f"Opening browser: {url}")
    webbrowser.open(url)


def main():
    print(r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║      ██╗   ██╗███████╗ ██████╗  █████╗  ██████╗  ██████╗ ██████╗ ║
    ║      ██║   ██║██╔════╝██╔════╝ ██╔══██╗██╔═══██╗██╔════╝██╔══██╗║
    ║      ██║   ██║█████╗  ██║  ███╗███████║██║   ██║██║     ██████╔╝║
    ║      ╚██╗ ██╔╝██╔══╝  ██║   ██║██╔══██║██║   ██║██║     ██╔══██╗║
    ║       ╚████╔╝ ███████╗╚██████╔╝██║  ██║╚██████╔╝╚██████╗██║  ██║║
    ║        ╚═══╝  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝║
    ║                                                           ║
    ║           ML-Powered Multi-Engine OCR Toolkit              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    logger.info(f"Starting VegaOCR at http://{config.HOST}:{config.PORT}")
    logger.info("Press Ctrl+C to stop")

    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
