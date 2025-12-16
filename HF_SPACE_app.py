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
# This function has a bug where it doesn't handle boolean schemas (True/False as schema values)
try:
    import gradio_client.utils as client_utils
    original_json_schema_to_python_type = client_utils.json_schema_to_python_type
    
    def patched_json_schema_to_python_type(schema):
        """Patched version that handles boolean schemas gracefully"""
        try:
            # Handle case where schema itself is a boolean
            if isinstance(schema, bool):
                return "Any"
            return original_json_schema_to_python_type(schema)
        except Exception:
            # If any error occurs in schema parsing, return a safe default
            return "Any"
    
    # Replace the function
    client_utils.json_schema_to_python_type = patched_json_schema_to_python_type
except (ImportError, AttributeError):
    pass

# Also monkey-patch the internal _json_schema_to_python_type function
try:
    import gradio_client.utils as client_utils
    if hasattr(client_utils, '_json_schema_to_python_type'):
        original_inner = client_utils._json_schema_to_python_type
        
        def patched_inner(schema, defs=None):
            """Patched inner function that gracefully handles all schema types"""
            try:
                # Handle boolean schemas (True/False)
                if isinstance(schema, bool):
                    return "Any"
                return original_inner(schema, defs)
            except Exception:
                # Return safe default on any error
                return "Any"
        
        client_utils._json_schema_to_python_type = patched_inner
except (ImportError, AttributeError):
    pass

# Add src directory to path so slgps package is importable
repo_root = Path(__file__).resolve().parent
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Monkey-patch Gradio to prevent API info generation errors
try:
    from gradio import blocks
    original_get_api_info = blocks.Blocks.get_api_info
    
    def patched_get_api_info(self):
        """Patched get_api_info that returns empty dict on error"""
        try:
            return original_get_api_info(self)
        except Exception as e:
            # Return minimal API info to prevent errors
            return {"components": {}, "dependent_ids": []}
    
    blocks.Blocks.get_api_info = patched_get_api_info
except (ImportError, AttributeError):
    pass

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
