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
    # Call a simple function with dummy data
    hrr = [0, 0.1, 0.5, 1.2, 0.8, 0.2, 0]
    threshold = 0.7
    ign_start, ign_end = utils.findIgnInterval(hrr, threshold)
    print(f"[DEBUG] app_minimal.py: findIgnInterval result: start={ign_start}, end={ign_end}", flush=True)
except Exception as e:
    print(f"[DEBUG] app_minimal.py: failed to import slgps.utils or call function: {e}", flush=True)

# Try importing TensorFlow
try:
    import tensorflow as tf
    print(f"[DEBUG] app_minimal.py: imported tensorflow version {tf.__version__}", flush=True)
except Exception as e:
    print(f"[DEBUG] app_minimal.py: failed to import tensorflow: {e}", flush=True)

# Try importing Cantera
try:
    import cantera as ct
    print(f"[DEBUG] app_minimal.py: imported cantera version {ct.__version__}", flush=True)
except Exception as e:
    print(f"[DEBUG] app_minimal.py: failed to import cantera: {e}", flush=True)

def echo(text):
    return f"You typed: {text}"

def gradio_find_ign_interval(hrr_str, threshold):
    try:
        hrr = [float(x.strip()) for x in hrr_str.split(",") if x.strip()]
        threshold = float(threshold)
        start, end = utils.findIgnInterval(hrr, threshold)
        return f"Ignition interval: start={start}, end={end}"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Minimal Gradio App Test")
    inp = gr.Textbox(label="Type something")
    out = gr.Textbox(label="Echo output")
    btn = gr.Button("Echo")
    btn.click(fn=echo, inputs=inp, outputs=out)

    gr.Markdown("## Test findIgnInterval from slgps.utils")
    hrr_input = gr.Textbox(label="HRR list (comma-separated)", value="0, 0.1, 0.5, 1.2, 0.8, 0.2, 0")
    threshold_input = gr.Textbox(label="Threshold", value="0.7")
    result_output = gr.Textbox(label="Result")
    test_btn = gr.Button("Run findIgnInterval")
    test_btn.click(fn=gradio_find_ign_interval, inputs=[hrr_input, threshold_input], outputs=result_output)

if __name__ == "__main__":
    print("[DEBUG] app_minimal.py: launching Gradio", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)
    print("[DEBUG] app_minimal.py: Gradio launch returned", flush=True)
