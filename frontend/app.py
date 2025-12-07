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
from typing import Tuple, Dict, Any
import traceback

try:
    from slgps.make_data_parallel import make_data_parallel
    from slgps.mech_train import make_model
except ImportError:
    print("Warning: SL-GPS not installed. Some features may not work.")


# Global state to track progress
app_state = {
    "data_path": None,
    "model_path": None,
    "scaler_path": None,
    "status": "Ready"
}


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
) -> Tuple[str, str]:
    """
    Generate training dataset from autoignition simulations with GPS.
    
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
        
    Returns:
        Tuple of (status_message, error_message)
    """
    try:
        # Create output directory
        data_dir = "generated_data"
        os.makedirs(data_dir, exist_ok=True)
        app_state["data_path"] = data_dir
        
        # Parse species ranges from JSON string
        try:
            species_ranges = json.loads(species_string) if species_string.strip() else {}
        except json.JSONDecodeError:
            return "", "Error: Invalid species ranges JSON format"
        
        # Save mechanism file to temp location
        mech_path = os.path.join(data_dir, "mechanism.cti")
        if mechanism_file is None:
            return "", "Error: No mechanism file uploaded"
        
        # Copy mechanism file
        shutil.copy(mechanism_file.name, mech_path)
        
        # Call data generation
        status = f"Generating dataset with {n_cases} simulations...\n"
        status += f"Fuel: {fuel_species}, T: {temp_min}-{temp_max}K, P: {pressure_min}-{pressure_max} log(atm)\n"
        status += f"GPS alpha: {alpha}, Always threshold: {always_threshold}, Never threshold: {never_threshold}\n"
        
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
        
        status += f"\n✅ Dataset generated successfully in '{data_dir}'\n"
        status += f"Output files:\n"
        status += f"  - {data_dir}/data.csv (state vectors)\n"
        status += f"  - {data_dir}/species.csv (species masks)\n"
        status += f"  - {data_dir}/always_spec_nums.csv\n"
        status += f"  - {data_dir}/never_spec_nums.csv\n"
        
        app_state["status"] = "Dataset generated"
        return status, ""
        
    except Exception as e:
        error_msg = f"Error generating dataset:\n{str(e)}\n\n{traceback.format_exc()}"
        return "", error_msg


def train_neural_network(
    input_species_string: str,
    n_hidden_layers: int,
    neurons_per_layer: int,
    learning_rate: float = 0.001,
) -> Tuple[str, str]:
    """
    Train neural network for species importance prediction.
    
    Args:
        input_species_string: JSON string of input species names
        n_hidden_layers: Number of hidden layers
        neurons_per_layer: Number of neurons per hidden layer
        learning_rate: Learning rate for training
        
    Returns:
        Tuple of (status_message, error_message)
    """
    try:
        if app_state["data_path"] is None:
            return "", "Error: Generate dataset first before training NN"
        
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
        
        status = f"Training neural network...\n"
        status += f"Hidden layers: {n_hidden_layers}\n"
        status += f"Neurons per layer: {neurons_per_layer}\n"
        status += f"Input species: {', '.join(input_specs)}\n"
        status += f"Data path: {data_path}\n\n"
        
        # Note: NN architecture customization requires modifying mech_train.py
        # For now, we'll use default architecture and show user how to customize
        make_model(
            input_specs=input_specs,
            data_path=data_path,
            scaler_path=scaler_path,
            model_path=model_path
        )
        
        app_state["model_path"] = model_path
        app_state["scaler_path"] = scaler_path
        
        status += f"\n✅ Neural network trained successfully!\n"
        status += f"Model saved to: {model_path}\n"
        status += f"Scaler saved to: {scaler_path}\n\n"
        status += f"⚠️ Note: Current implementation uses default architecture (16 neurons, 1 hidden layer).\n"
        status += f"To customize layers and neurons, edit src/slgps/mech_train.py::spec_train() function.\n"
        
        app_state["status"] = "NN trained"
        return status, ""
        
    except Exception as e:
        error_msg = f"Error training neural network:\n{str(e)}\n\n{traceback.format_exc()}"
        return "", error_msg


def create_gradio_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="SL-GPS Chemistry Reduction GUI", theme=gr.themes.Soft()) as demo:
        
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
            
            # Callbacks
            gen_button.click(
                fn=generate_dataset,
                inputs=[
                    mechanism_file, fuel_species, n_cases,
                    temp_min, temp_max, pressure_min, pressure_max,
                    alpha, always_threshold, never_threshold, species_string
                ],
                outputs=[gen_status, gen_error]
            )
            
            def clear_gen():
                return None, "", ""
            
            clear_button.click(
                fn=clear_gen,
                outputs=[mechanism_file, gen_status, gen_error]
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
            
            # Callback
            train_button.click(
                fn=train_neural_network,
                inputs=[input_species_string, n_hidden_layers, neurons_per_layer, learning_rate],
                outputs=[train_status, train_error]
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
