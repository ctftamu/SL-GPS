#!/usr/bin/env python3
"""
Main entry point for SL-GPS Gradio application.
This file is required by Hugging Face Spaces.

Uses lazy imports to minimize startup time on HF Spaces.
"""

if __name__ == "__main__":
    # Import Gradio only when running
    import gradio as gr
    from frontend.app import create_gradio_interface
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.launch(share=False, show_error=True, theme=gr.themes.Soft())
