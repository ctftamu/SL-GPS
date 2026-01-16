#!/usr/bin/env python3
print("[DEBUG] app.py: container startup", flush=True)
"""
Main entry point for SL-GPS Gradio application.
This file is required by Hugging Face Spaces.

Minimal initialization - all heavy imports deferred until access.
"""


# Ensure src/ is on path for import
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import gradio as gr
from frontend.app import create_gradio_interface

# Expose demo at module level for HF Spaces
demo = create_gradio_interface()
print("[DEBUG] app.py: demo assigned", flush=True)

if __name__ == "__main__":
    print("[DEBUG] app.py: entered __main__ block", flush=True)
    print("[DEBUG] app.py: launching Gradio", flush=True)
    demo.launch(share=False, show_error=True, theme=gr.themes.Soft())
    print("[DEBUG] app.py: Gradio launch returned", flush=True)



