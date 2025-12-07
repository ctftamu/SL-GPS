"""
Launch the SL-GPS Gradio application.

Run this with: python -m frontend
"""

from frontend.app import create_gradio_interface

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=False, show_error=True)
