"""
SL-GPS GUI Application using Gradio

A browser-based interface for:
1. Data generation from autoignition simulations with GPS
2. Neural network training for species importance prediction
3. Configuration of mechanism reduction parameters
"""

import gradio as gr
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Generator
import traceback
import sys
import io
import contextlib
from datetime import datetime
import threading
import queue

# Ensure local `src/` is on sys.path so `slgps` package imports work when running
# the frontend from the repository root (e.g. `python -m frontend`). This helps
# avoid NameError when backend functions aren't importable.
try:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
except Exception:
    pass

# Backend placeholders (will be overwritten if imports succeed)
make_data_parallel = None
make_model = None

try:
    from slgps.make_data_parallel import make_data_parallel
    from slgps.mech_train import make_model
except Exception:
    # Print a helpful warning — the GUI can still start, but dataset generation
    # and training features will return a clear error message in the UI.
    print("Warning: SL-GPS backend not importable. Dataset generation and training will be disabled in the GUI.")


# Global state to track progress
app_state = {
    "data_path": None,
    "model_path": None,
    "scaler_path": None,
    "status": "Ready",
    "log_buffer": []
}


class LogCapture:
    """Capture stderr/stdout and store in log queue for real-time streaming"""
    def __init__(self, log_queue=None):
        self.original_stderr = sys.stderr
        self.original_stdout = sys.stdout
        self.log_queue = log_queue
        
    def __enter__(self):
        sys.stderr = self
        sys.stdout = self
        return self
        
    def __exit__(self, *args):
        sys.stderr = self.original_stderr
        sys.stdout = self.original_stdout
        
    def write(self, message):
        if message.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_msg = f"[{timestamp}] {message}"
            # Store in app state for reference
            app_state["log_buffer"].append(log_msg)
            # Send to queue if provided (for real-time streaming)
            if self.log_queue:
                try:
                    self.log_queue.put(log_msg, block=False)
                except queue.Full:
                    pass
            # Also write to original stderr for server logs
            self.original_stderr.write(log_msg + "\n")
            
    def flush(self):
        pass


def generate_dataset(
    mechanism_file,
    fuel_species: str,
    n_cases: int,
    temp_min: float,
    temp_max: float,
    pressure_min: float,
    pressure_max: float,
    alpha: float,
    always_threshold: float,
    never_threshold: float,
    species_string: str,
    progress=gr.Progress(),
) -> Generator[Tuple[str, str], None, None]:
    """
    Generate training dataset from autoignition simulations with GPS.
    Streams output in real-time to the UI.
    
    Args:
        mechanism_file: Uploaded Cantera CTI file
        fuel_species: Name of fuel species (e.g., 'CH4')
        n_cases: Number of autoignition simulations
        temp_min/temp_max: Temperature range (K)
        pressure_min/pressure_max: Pressure range (log10 atm)
        alpha: GPS pathway threshold
        always_threshold: Species occurrence threshold for "always include"
        never_threshold: Species occurrence threshold for "never include"
        species_string: JSON string of species ranges
        progress: Gradio progress object
        
    Yields:
        Tuple of (status_message, error_message)
    """
    try:
        # Clear log buffer and setup log queue
        app_state["log_buffer"] = []
        log_queue = queue.Queue(maxsize=100)
        accumulated_output = f"[{datetime.now().strftime('%H:%M:%S')}] Starting dataset generation...\n"
        
        yield accumulated_output, ""
        
        # Create output directory
        data_dir = "generated_data"
        os.makedirs(data_dir, exist_ok=True)
        app_state["data_path"] = data_dir
        
        # Parse species ranges from JSON string
        try:
            species_ranges = json.loads(species_string) if species_string.strip() else {}
        except json.JSONDecodeError:
            yield "", "Error: Invalid species ranges JSON format"
            return
        
        # Save mechanism file to temp location
        mech_path = os.path.join(data_dir, "mechanism.cti")
        if mechanism_file is None:
            yield "", "Error: No mechanism file uploaded"
            return
        
        # Copy mechanism file
        shutil.copy(mechanism_file.name, mech_path)
        accumulated_output += f"[{datetime.now().strftime('%H:%M:%S')}] Mechanism file saved to {mech_path}\n"
        yield accumulated_output, ""
        
        # Guard: ensure backend function is available
        if make_data_parallel is None:
            yield "", (
                "Error: backend `make_data_parallel` not available.\n"
                "Run the frontend from the repository root so the `src/` package is importable, "
                "or install the package (e.g. `pip install -e .`)."
            )
            return

        # Call data generation with log capture
        accumulated_output += f"Generating dataset with {n_cases} simulations...\n"
        accumulated_output += f"Fuel: {fuel_species}, T: {temp_min}-{temp_max}K, P: {pressure_min}-{pressure_max} log(atm)\n"
        accumulated_output += f"GPS alpha: {alpha}, Always threshold: {always_threshold}, Never threshold: {never_threshold}\n\n"
        yield accumulated_output, ""
        
        progress(0, desc="Initializing...")
        
        # Run backend function in a thread and stream logs in real-time
        error_holder = {"error": None}
        
        def run_backend():
            try:
                with LogCapture(log_queue):
                    make_data_parallel(
                        fuel=fuel_species,
                        mech_file=mech_path,
                        end_threshold=2e5,
                        ign_HRR_threshold_div=300,
                        ign_GPS_resolution=200,
                        norm_GPS_resolution=40,
                        GPS_per_interval=4,
                        n_cases=n_cases,
                        t_rng=[temp_min, temp_max],
                        p_rng=[pressure_min, pressure_max],
                        phi_rng=[0.6, 1.4],  # Not used if species_ranges is set
                        alpha=alpha,
                        always_threshold=always_threshold,
                        never_threshold=never_threshold,
                        pathname=data_dir,
                        species_ranges=species_ranges
                    )
            except Exception as e:
                error_holder["error"] = e
        
        # Start backend in thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Stream logs while backend runs (optimized polling)
        import time
        last_yield_time = time.time()
        while backend_thread.is_alive():
            # Check for logs
            has_logs = False
            while True:
                try:
                    log_msg = log_queue.get_nowait()
                    accumulated_output += log_msg + "\n"
                    has_logs = True
                except queue.Empty:
                    break
            
            # Yield only if we got new logs and at least 1 second has passed
            # This reduces UI update frequency and improves performance
            current_time = time.time()
            if has_logs and (current_time - last_yield_time) >= 1.0:
                yield accumulated_output, ""
                last_yield_time = current_time
            
            # Increased sleep interval to reduce CPU usage (was 0.1s)
            time.sleep(0.5)
        
        # Wait for thread to complete
        backend_thread.join(timeout=5)
        
        # Collect any final logs
        while True:
            try:
                log_msg = log_queue.get_nowait()
                accumulated_output += log_msg + "\n"
            except queue.Empty:
                break
        
        # Check for errors
        if error_holder["error"]:
            accumulated_output += f"\n❌ ERROR during execution: {str(error_holder['error'])}\n"
            accumulated_output += f"Full traceback:\n{traceback.format_exc()}\n"
            yield "", accumulated_output
            return
        
        accumulated_output += f"\n✅ Dataset generated successfully in '{data_dir}'\n"
        accumulated_output += f"Output files:\n"
        accumulated_output += f"  - {data_dir}/data.csv (state vectors)\n"
        accumulated_output += f"  - {data_dir}/species.csv (species masks)\n"
        accumulated_output += f"  - {data_dir}/always_spec_nums.csv\n"
        accumulated_output += f"  - {data_dir}/never_spec_nums.csv\n"
        
        # Show available species for user reference
        try:
            import pandas as pd
            data_df = pd.read_csv(os.path.join(data_dir, "data.csv"))
            available_species = [col for col in data_df.columns if col not in ['# Temperature', 'Atmospheres']]
            accumulated_output += f"\n📊 Available species for training ({len(available_species)} total):\n"
            accumulated_output += f"  {', '.join(sorted(available_species))}\n"
            accumulated_output += f"\nℹ️  Use these species names in the Neural Network Training tab.\n"
        except Exception as e:
            accumulated_output += f"\n⚠️  Could not read available species: {str(e)}\n"
        
        progress(1, desc="Complete!")
        app_state["status"] = "Dataset generated"
        yield accumulated_output, ""
        
    except Exception as e:
        error_msg = f"❌ Error generating dataset:\n{str(e)}\n\n"
        error_msg += f"Full traceback:\n{traceback.format_exc()}"
        yield "", error_msg



def train_neural_network(
    input_species_string: str,
    n_hidden_layers: int,
    neurons_per_layer: int,
    learning_rate: float = 0.001,
    num_processes: int = 1,
    progress=gr.Progress(),
) -> Generator[Tuple[str, str], None, None]:
    """
    Train neural network for species importance prediction.
    Streams output in real-time to the UI.
    
    Args:
        input_species_string: JSON string of input species names
        n_hidden_layers: Number of hidden layers
        neurons_per_layer: Number of neurons per hidden layer
        learning_rate: Learning rate for training
        num_processes: Number of parallel processes
        progress: Gradio progress object
        
    Yields:
        Tuple of (status_message, error_message)
    """
    try:
        # Clear log buffer and setup log queue
        app_state["log_buffer"] = []
        log_queue = queue.Queue(maxsize=100)
        accumulated_output = f"[{datetime.now().strftime('%H:%M:%S')}] Starting neural network training...\n"
        
        yield accumulated_output, ""
        
        if app_state["data_path"] is None:
            yield "", "Error: Generate dataset first before training NN"
            return
        
        # Parse input species
        try:
            input_specs = json.loads(input_species_string)
            if not isinstance(input_specs, list):
                input_specs = [s.strip() for s in input_species_string.split(",")]
        except:
            input_specs = [s.strip() for s in input_species_string.split(",")]
        
        data_path = app_state["data_path"]
        scaler_path = os.path.join(data_path, "scaler.pkl")
        model_path = os.path.join(data_path, "model.h5")
        
        # Validate that input species exist in the training data
        data_csv_path = os.path.join(data_path, "data.csv")
        try:
            import pandas as pd
            data_df = pd.read_csv(data_csv_path)
            available_species = [col for col in data_df.columns if col not in ['# Temperature', 'Atmospheres']]
            
            # Check for missing species
            missing_species = [sp for sp in input_specs if sp not in available_species]
            if missing_species:
                error_msg = f"❌ Error: The following species were not found in the training data:\n"
                error_msg += f"  Missing: {', '.join(missing_species)}\n\n"
                error_msg += f"Available species in your data ({len(available_species)} total):\n"
                error_msg += f"  {', '.join(sorted(available_species))}\n\n"
                error_msg += f"Please update the input species list to use only species from the data.\n"
                yield "", error_msg
                return
        except Exception as e:
            yield "", f"Error reading training data: {str(e)}"
            return
        
        accumulated_output += f"Training neural network...\n"
        accumulated_output += f"Hidden layers: {n_hidden_layers}\n"
        accumulated_output += f"Neurons per layer: {neurons_per_layer}\n"
        accumulated_output += f"Num processes: {num_processes}\n"
        accumulated_output += f"Input species: {', '.join(input_specs)}\n"
        accumulated_output += f"Data path: {data_path}\n\n"
        
        yield accumulated_output, ""
        progress(0, desc="Initializing...")
        
        # Run backend function in a thread and stream logs in real-time
        error_holder = {"error": None}
        
        def run_backend():
            try:
                with LogCapture(log_queue):
                    make_model(
                        input_specs=input_specs,
                        data_path=data_path,
                        scaler_path=scaler_path,
                        model_path=model_path,
                        num_hidden_layers=int(n_hidden_layers),
                        neurons_per_layer=int(neurons_per_layer),
                        num_processes=max(1, int(num_processes))
                    )
            except Exception as e:
                error_holder["error"] = e
        
        # Start backend in thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Stream logs while backend runs (optimized polling)
        import time
        last_yield_time = time.time()
        while backend_thread.is_alive():
            # Check for logs
            has_logs = False
            while True:
                try:
                    log_msg = log_queue.get_nowait()
                    accumulated_output += log_msg + "\n"
                    has_logs = True
                except queue.Empty:
                    break
            
            # Yield only if we got new logs and at least 1 second has passed
            # This reduces UI update frequency and improves performance
            current_time = time.time()
            if has_logs and (current_time - last_yield_time) >= 1.0:
                yield accumulated_output, ""
                last_yield_time = current_time
            
            # Increased sleep interval to reduce CPU usage (was 0.1s)
            time.sleep(0.5)
        
        # Wait for thread to complete
        backend_thread.join(timeout=5)
        
        # Collect any final logs
        while True:
            try:
                log_msg = log_queue.get_nowait()
                accumulated_output += log_msg + "\n"
            except queue.Empty:
                break
        
        # Check for errors
        if error_holder["error"]:
            accumulated_output += f"\n❌ ERROR during training: {str(error_holder['error'])}\n"
            accumulated_output += f"Full traceback:\n{traceback.format_exc()}\n"
            yield "", accumulated_output
            return
        
        app_state["model_path"] = model_path
        app_state["scaler_path"] = scaler_path
        
        accumulated_output += f"\n✅ Neural network trained successfully!\n"
        accumulated_output += f"Model saved to: {model_path}\n"
        accumulated_output += f"Scaler saved to: {scaler_path}\n\n"
        accumulated_output += f"✅ Architecture applied: {int(n_hidden_layers)} hidden layers, {int(neurons_per_layer)} neurons/layer.\n"
        accumulated_output += f"✅ Training processes used: {int(num_processes)}\n"
        
        progress(1, desc="Complete!")
        app_state["status"] = "NN trained"
        yield accumulated_output, ""
        
    except Exception as e:
        error_msg = f"❌ Error training neural network:\n{str(e)}\n\n"
        error_msg += f"Full traceback:\n{traceback.format_exc()}"
        yield "", error_msg


def create_gradio_interface():
    """Create and return the Gradio interface."""
    
    # Create a Blocks context. Newer Gradio versions accept a `theme` argument,
    # older versions do not — handle both cases so the frontend works across
    # multiple Gradio releases.
    try:
        demo_ctx = gr.Blocks(title="SL-GPS Chemistry Reduction GUI", theme=gr.themes.Soft())
    except TypeError:
        demo_ctx = gr.Blocks(title="SL-GPS Chemistry Reduction GUI")

    with demo_ctx as demo:
        gr.Markdown("# 🧪 SL-GPS Chemistry Reduction GUI")
        gr.Markdown(
            """
            Automated neural network-based chemistry reduction using Global Pathway Selection (GPS).
            
            **Workflow:**
            1. Upload a Cantera mechanism file (CTI format)
            2. Configure simulation parameters
            3. Generate training dataset
            4. Train neural network
            5. Download results
            """
        )
        
        # ============================================================================
        # TAB 1: DATASET GENERATION
        # ============================================================================
        with gr.Tab("📊 Generate Dataset"):
            
            gr.Markdown("### Step 1: Upload Mechanism and Configure Simulation")
            
            with gr.Row():
                mechanism_file = gr.File(
                    label="Upload Cantera Mechanism (.cti)",
                    file_types=[".cti"],
                    type="filepath"
                )
            
            with gr.Row():
                fuel_species = gr.Textbox(
                    label="Fuel Species Name",
                    value="CH4",
                    placeholder="e.g., CH4, H2, C2H4"
                )
                n_cases = gr.Number(
                    label="Number of Simulations",
                    value=100,
                    precision=0
                )
            
            gr.Markdown("#### Temperature Range")
            with gr.Row():
                temp_min = gr.Slider(
                    label="Min Temperature (K)",
                    minimum=300,
                    maximum=3000,
                    value=800,
                    step=100
                )
                temp_max = gr.Slider(
                    label="Max Temperature (K)",
                    minimum=300,
                    maximum=3000,
                    value=2300,
                    step=100
                )
            
            gr.Markdown("#### Pressure Range")
            with gr.Row():
                pressure_min = gr.Slider(
                    label="Min Pressure (log10 atm)",
                    minimum=0,
                    maximum=5,
                    value=2.1,
                    step=0.1
                )
                pressure_max = gr.Slider(
                    label="Max Pressure (log10 atm)",
                    minimum=0,
                    maximum=5,
                    value=2.5,
                    step=0.1
                )
            
            gr.Markdown("#### GPS Parameters")
            with gr.Row():
                alpha = gr.Slider(
                    label="GPS Alpha (pathway threshold)",
                    minimum=0.0001,
                    maximum=0.1,
                    value=0.001,
                    step=0.0001,
                    scale=1
                )
            
            with gr.Row():
                always_threshold = gr.Slider(
                    label="Always Include Threshold",
                    minimum=0.5,
                    maximum=1.0,
                    value=0.99,
                    step=0.01,
                    info="Species in >X% of simulations are always included"
                )
                never_threshold = gr.Slider(
                    label="Never Include Threshold",
                    minimum=0.0,
                    maximum=0.5,
                    value=0.01,
                    step=0.01,
                    info="Species in <X% of simulations are never included"
                )
            
            gr.Markdown("#### Species Composition Ranges (JSON)")
            species_string = gr.Textbox(
                label="Species Ranges (JSON format)",
                value='{"CH4": [0, 1], "O2": [0, 0.4], "N2": [0, 0.8]}',
                lines=4,
                placeholder='{"CH4": [0, 1], "O2": [0, 0.4], "N2": [0, 0.8], "CO2": [0, 0.005]}',
                info="Define min/max mole fraction for each species"
            )
            
            # Generate button
            with gr.Row():
                gen_button = gr.Button("🚀 Generate Dataset", variant="primary", scale=2)
                clear_button = gr.Button("🔄 Clear", scale=1)
            
            # Output
            with gr.Row():
                gen_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=10,
                    max_lines=15
                )
            
            gen_error = gr.Textbox(
                label="Error (if any)",
                interactive=False,
                lines=5,
                visible=True
            )
            
            # Download buttons for dataset
            gr.Markdown("#### Download Dataset Files")
            with gr.Row():
                with gr.Column():
                    gr.Label(value="Data Files")
                    data_csv_file = gr.File(
                        label="data.csv",
                        type="filepath",
                        interactive=False
                    )
                with gr.Column():
                    gr.Label(value="Species Files")
                    species_csv_file = gr.File(
                        label="species.csv",
                        type="filepath",
                        interactive=False
                    )
            
            with gr.Row():
                always_spec_file = gr.File(
                    label="always_spec_nums.csv",
                    type="filepath",
                    interactive=False,
                    scale=1
                )
                never_spec_file = gr.File(
                    label="never_spec_nums.csv",
                    type="filepath",
                    interactive=False,
                    scale=1
                )
                var_spec_file = gr.File(
                    label="var_spec_nums.csv",
                    type="filepath",
                    interactive=False,
                    scale=1
                )
            
            # Callbacks
            def update_dataset_downloads():
                """Return all dataset files if generation succeeded"""
                data_path = app_state.get("data_path")
                if not data_path:
                    return None, None, None, None, None
                
                files = {
                    "data_csv": os.path.join(data_path, "data.csv"),
                    "species_csv": os.path.join(data_path, "species.csv"),
                    "always_spec": os.path.join(data_path, "always_spec_nums.csv"),
                    "never_spec": os.path.join(data_path, "never_spec_nums.csv"),
                    "var_spec": os.path.join(data_path, "var_spec_nums.csv"),
                }
                
                return tuple(
                    f if os.path.exists(f) else None 
                    for f in files.values()
                )
            
            gen_button.click(
                fn=generate_dataset,
                inputs=[
                    mechanism_file, fuel_species, n_cases,
                    temp_min, temp_max, pressure_min, pressure_max,
                    alpha, always_threshold, never_threshold, species_string
                ],
                outputs=[gen_status, gen_error],
                concurrency_limit=1
            )
            
            # Update downloads after dataset generation
            gen_status.change(
                fn=update_dataset_downloads,
                outputs=[data_csv_file, species_csv_file, always_spec_file, never_spec_file, var_spec_file]
            )
            
            def clear_gen():
                return None, "", "", None, None, None, None, None
            
            clear_button.click(
                fn=clear_gen,
                outputs=[mechanism_file, gen_status, gen_error, data_csv_file, species_csv_file, always_spec_file, never_spec_file, var_spec_file]
            )
        
        # ============================================================================
        # TAB 2: NEURAL NETWORK TRAINING
        # ============================================================================
        with gr.Tab("🧠 Train Neural Network"):
            
            gr.Markdown("### Step 2: Configure and Train Neural Network")
            gr.Markdown(
                "Generate a dataset first using the **Generate Dataset** tab. "
                "Then configure your neural network architecture below."
            )
            
            gr.Markdown("#### Input Species Selection")
            input_species_string = gr.Textbox(
                label="Input Species (comma-separated or JSON list)",
                value="CH4, H2O, OH, H, CO, O2, CO2, O, CH3, CH, H2",
                lines=3,
                placeholder="Species to use as ANN inputs (must match training data)"
            )
            
            gr.Markdown("#### Neural Network Architecture")
            with gr.Row():
                n_hidden_layers = gr.Slider(
                    label="Number of Hidden Layers",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    info="Deeper networks can capture more complex patterns"
                )
                neurons_per_layer = gr.Slider(
                    label="Neurons per Hidden Layer",
                    minimum=4,
                    maximum=256,
                    value=16,
                    step=4,
                    info="More neurons = more capacity but slower training"
                )
            
            gr.Markdown("#### Training Settings")
            learning_rate = gr.Slider(
                label="Learning Rate",
                minimum=0.0001,
                maximum=0.1,
                value=0.001,
                step=0.0001,
                info="How fast the network learns (smaller = slower but more stable)"
            )

            num_processes = gr.Slider(
                label="Number of Processes",
                minimum=1,
                maximum=64,
                value=1,
                step=1,
                info="Parallel training workers (set to 1 for sequential / debugging)"
            )
            
            gr.Markdown(
                """
                ⚠️ **Important:** The current implementation uses a default architecture. 
                To apply your custom layer/neuron settings, you'll need to:
                
                1. Edit `src/slgps/mech_train.py`
                2. Modify the `spec_train()` function to use your parameters
                3. See API Reference documentation for details
                """
            )
            
            # Train button
            with gr.Row():
                train_button = gr.Button("🚀 Train Neural Network", variant="primary", scale=2)
            
            # Output
            train_status = gr.Textbox(
                label="Training Status",
                interactive=False,
                lines=10,
                max_lines=15
            )
            
            train_error = gr.Textbox(
                label="Error (if any)",
                interactive=False,
                lines=5,
                visible=True
            )
            
            # Download buttons
            gr.Markdown("#### Download Trained Model")
            with gr.Row():
                model_file = gr.File(
                    label="model.h5",
                    type="filepath",
                    interactive=False
                )
                scaler_file = gr.File(
                    label="scaler.pkl",
                    type="filepath",
                    interactive=False
                )
            
            # Callback
            def update_downloads():
                """Return download files if training succeeded"""
                model_path = app_state.get("model_path")
                scaler_path = app_state.get("scaler_path")
                return (
                    model_path if model_path and os.path.exists(model_path) else None,
                    scaler_path if scaler_path and os.path.exists(scaler_path) else None
                )
            
            train_button.click(
                fn=train_neural_network,
                inputs=[input_species_string, n_hidden_layers, neurons_per_layer, learning_rate, num_processes],
                outputs=[train_status, train_error],
                concurrency_limit=1
            )
            
            # Update downloads after training
            train_status.change(
                fn=update_downloads,
                outputs=[model_file, scaler_file]
            )
        
        # ============================================================================
        # TAB 3: DOCUMENTATION
        # ============================================================================
        with gr.Tab("📖 Documentation"):
            
            gr.Markdown(
                """
                # SL-GPS: Supervised Learning - Global Pathway Selection
                
                ## What is this?
                
                SL-GPS is a framework for **automated chemistry reduction** in combustion simulations. 
                It trains neural networks to predict which chemical species are important at different 
                points in a simulation, enabling dynamic mechanism reduction.
                
                ## Workflow
                
                1. **Data Generation**: Run multiple autoignition simulations with detailed mechanisms. 
                   Use GPS (Global Pathway Selection) to identify important species in each timestep.
                   
                2. **ANN Training**: Train a neural network to predict important species based on 
                   thermochemical state (temperature, pressure, composition).
                   
                3. **Adaptive Simulation**: Use the trained ANN during new simulations to dynamically 
                   reduce mechanisms in real-time.
                
                ## Key Parameters
                
                ### Dataset Generation
                - **Fuel Species**: The main reactant (e.g., CH4, H2)
                - **Number of Cases**: More simulations = better training data but slower
                - **Temperature/Pressure Range**: Initial conditions for simulations
                - **GPS Alpha**: Controls pathway importance threshold (smaller = more species)
                - **Always/Never Thresholds**: Species that appear frequently (always) or rarely (never)
                
                ### Neural Network
                - **Input Species**: Which species affect the ANN decision
                - **Hidden Layers**: More layers = more model complexity
                - **Neurons per Layer**: More neurons = larger model capacity
                
                ## Output Files
                
                After dataset generation, you'll have:
                - `data.csv`: State vectors (temperature, pressure, compositions)
                - `species.csv`: Binary masks indicating important species
                - `always_spec_nums.csv`: Species always included (99%+ occurrence)
                - `never_spec_nums.csv`: Species never included (<1% occurrence)
                
                After NN training:
                - `model.h5`: Trained Keras neural network
                - `scaler.pkl`: Input normalization scaler
                
                ## For More Information
                
                - Full documentation: https://ctftamu.github.io/SL-GPS/
                - GitHub repository: https://github.com/ctftamu/SL-GPS
                - Paper: [Mishra et al., 2022](https://doi.org/10.1016/j.combustflame.2022.112279)
                
                ## Contact
                
                - Rohit Mishra: rmishra@tamu.edu
                - Aaron Nelson: aaronnelson@tamu.edu
                """
            )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=False, show_error=True)
