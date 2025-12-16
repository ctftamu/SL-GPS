#!/usr/bin/env python3
"""
SL-GPS GUI Application for Hugging Face Spaces

This is the entry point for the Gradio application running on HF Spaces.
It imports and runs the main frontend application from the SL-GPS package.
"""

import sys
from pathlib import Path

# Add src directory to path so slgps package is importable
repo_root = Path(__file__).resolve().parent
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import and launch the main frontend application
from frontend.app import app

if __name__ == "__main__":
    app.launch()
