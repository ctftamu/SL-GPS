#!/usr/bin/env python3
"""
Main entry point for SL-GPS Gradio application.
This file is required by Hugging Face Spaces.

It imports and launches the actual Gradio interface from frontend/app.py
"""

from frontend.app import create_gradio_interface

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=False, show_error=True)
