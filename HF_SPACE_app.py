#!/usr/bin/env python3
"""
SL-GPS GUI Application for Hugging Face Spaces

This is the entry point for the Gradio application running on HF Spaces.
It imports and runs the main frontend application from the SL-GPS package.
"""

import sys
from pathlib import Path

# Monkey-patch huggingface_hub to add HfFolder back (removed in newer versions)
# This is needed for Gradio compatibility
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'HfFolder'):
        # Create a dummy HfFolder class for compatibility
        class HfFolder:
            @staticmethod
            def get_subfolder(subfolder):
                from pathlib import Path
                return Path.home() / '.cache' / 'huggingface' / subfolder
            
            @staticmethod
            def save_token(token):
                pass
        
        huggingface_hub.HfFolder = HfFolder
except ImportError:
    pass

# Monkey-patch gradio_client to fix json_schema_to_python_type error
# This function has a bug where it doesn't handle all schema types properly
try:
    import gradio_client.utils as client_utils
    original_get_type = None
    
    def patched_get_type(schema):
        """Patched version that handles bool schemas gracefully"""
        if isinstance(schema, bool):
            # If schema is a boolean (e.g., True or False), return a generic type
            return "Any"
        
        # Call original logic for non-bool schemas
        if original_get_type:
            return original_get_type(schema)
        return "Any"
    
    # Store original and replace
    if hasattr(client_utils, 'get_type'):
        original_get_type = client_utils.get_type
        client_utils.get_type = patched_get_type
except (ImportError, AttributeError):
    pass

# Add src directory to path so slgps package is importable
repo_root = Path(__file__).resolve().parent
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import and launch the Gradio interface
from frontend.app import create_gradio_interface

# Create and launch the app
demo = create_gradio_interface()
app = demo  # For HF Spaces compatibility

if __name__ == "__main__":
    demo.launch(share=False, show_error=True)
else:
    # For HuggingFace Spaces - disable problematic API info endpoint
    app = demo
    # Disable the /api endpoint that causes issues with json_schema_to_python_type
    try:
        app.config.enable_api = False
    except:
        pass
    app.config.enable_api = False
