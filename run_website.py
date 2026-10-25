"""
1-Click Launcher for AURA Vision Liquid Glass Face Recognition Platform
Starts the FastAPI server and opens your default browser automatically.
"""

import os
import sys
import webbrowser
import threading
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")


def main():
    print("=" * 65)
    print("       AURA VISION — LIQUID GLASS BIOMETRIC PLATFORM")
    print("=" * 65)
    print("[*] Starting backend server on http://localhost:8000")
    print("[*] Launching default web browser...")

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
