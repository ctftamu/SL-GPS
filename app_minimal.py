import gradio as gr
import sys
from pathlib import Path

# Add src/ to sys.path for imports
repo_root = Path(__file__).resolve().parent
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("[DEBUG] app_minimal.py: container startup", flush=True)

# Try importing a utility from main app
try:
    from slgps import utils
    print("[DEBUG] app_minimal.py: imported slgps.utils successfully", flush=True)
except Exception as e:
    print(f"[DEBUG] app_minimal.py: failed to import slgps.utils: {e}", flush=True)

def echo(text):
    return f"You typed: {text}"

with gr.Blocks() as demo:
    gr.Markdown("# Minimal Gradio App Test")
    inp = gr.Textbox(label="Type something")
    out = gr.Textbox(label="Echo output")
    btn = gr.Button("Echo")
    btn.click(fn=echo, inputs=inp, outputs=out)

if __name__ == "__main__":
    print("[DEBUG] app_minimal.py: launching Gradio", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)
    print("[DEBUG] app_minimal.py: Gradio launch returned", flush=True)
