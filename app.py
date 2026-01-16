#!/usr/bin/env python3
"""
Main entry point for SL-GPS Gradio application.
This file is required by Hugging Face Spaces.

Minimal initialization - all heavy imports deferred until access.
"""

# Global demo variable for HF Spaces to find
demo = None

def _create_demo():
    """Create the Gradio demo (lazy initialization)"""
    import sys
    from pathlib import Path
    
    # Ensure src/ is on path
    repo_root = Path(__file__).resolve().parent
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Now import Gradio and create interface
    import gradio as gr
    from frontend.app import create_gradio_interface
    
    return create_gradio_interface()

# HF Spaces will look for 'demo' at module level
# Use __getattr__ for lazy evaluation
def __getattr__(name):
    global demo
    if name == "demo":
        if demo is None:
            demo = _create_demo()
        return demo
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if __name__ == "__main__":
    # Only launch when running directly
    import gradio as gr
    app = _create_demo()
    app.launch(share=False, show_error=True, theme=gr.themes.Soft())



