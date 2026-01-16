import gradio as gr

print("[DEBUG] app_minimal.py: container startup", flush=True)

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
