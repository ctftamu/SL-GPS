#!/usr/bin/env python3
"""
Main entry point for SL-GPS Gradio application.
This file is required by Hugging Face Spaces.

Minimal initialization - all heavy imports deferred until launch.
"""

# Global demo variable for HF Spaces to find
demo = None

def launch_app():
    """Initialize and launch the Gradio app"""
    global demo
    
    # Import only when launching
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
    
    demo = create_gradio_interface()
    demo.launch(share=False, show_error=True, theme=gr.themes.Soft())
    return demo

# HF Spaces will look for 'demo' at module level
# Use __getattr__ for lazy evaluation
def __getattr__(name):
    if name == "demo":
        if globals()["demo"] is None:
            return launch_app()
        return globals()["demo"]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

if __name__ == "__main__":
    launch_app()


