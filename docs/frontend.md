# Frontend GUI

SL-GPS includes a beautiful browser-based GUI built with Gradio that makes it easy for non-technical users to generate datasets and train neural networks.

## 🌐 Try Online

**[Launch SL-GPS GUI on Hugging Face Spaces](https://huggingface.co/spaces/rmishra4/SL-GPS)** - No installation required! The frontend is fully deployed and ready to use.

<iframe
  src="https://huggingface.co/spaces/rmishra4/SL-GPS/embed"
  frameborder="0"
  width="100%"
  height="1200"
  style="border-radius: 8px; margin: 20px 0;"
></iframe>

---

## Quick Start

### Local Installation

```bash
# Install frontend dependencies
pip install -r frontend/requirements.txt
```

### Launch the GUI

```bash
# From SL-GPS root directory
python -m frontend
```

The GUI opens automatically in your default browser at `http://localhost:7860`

## Features

✨ **User-Friendly Interface**
- Intuitive tabs for each workflow step
- Sliders and input fields for all parameters
- Real-time status updates

📁 **File Management**
- Upload Cantera mechanism files directly
- Automatic output file organization
- Download results easily

⚙️ **Full Control**
- Configure all simulation parameters
- Customize neural network architecture
- Monitor training progress

🌐 **Web-Based**
- No installation needed beyond Python
- Share via public links
- Works on any device with a browser

## Workflow

### Tab 1: Generate Dataset

1. **Upload Mechanism File** (.cti format)
   - Examples: `gri30.cti`, `nHeptane.cti`

2. **Configure Simulation Parameters**
   - **Fuel Species**: Main reactant (e.g., CH4, H2)
   - **Number of Simulations**: How many autoignition cases to run
   - **Temperature Range**: Initial temperatures (K)
   - **Pressure Range**: Initial pressures (log₁₀ atm)

3. **Set GPS Parameters**
   - **Alpha**: Pathway importance threshold (smaller = more species)
   - **Always/Never Thresholds**: Frequency cutoffs for species inclusion

4. **Define Species Composition**
   - JSON format specifying min/max mole fractions
   - Example: `{"CH4": [0, 1], "O2": [0, 0.4]}`

5. **Click Generate**
   - Generates `generated_data/` folder with:
     - `data.csv` - State vectors
     - `species.csv` - Species importance masks
     - `always_spec_nums.csv` - Always-included species
     - `never_spec_nums.csv` - Never-included species

### Tab 2: Train Neural Network

1. **Specify Input Species**
   - Which species affect ANN decisions
   - Comma-separated list or JSON array

2. **Configure Architecture**
   - **Hidden Layers**: Depth of network (1-5)
   - **Neurons per Layer**: Capacity (4-256)
   - **Learning Rate**: Training speed

3. **Click Train**
   - Trains ensemble of ANNs in parallel
   - **Number of Processes**: Use the new slider to control parallel training workers. Set to `1` for sequential runs or debugging, higher values (e.g., 8, 16, 28) to use more CPU cores. The backend `make_model(..., num_processes=...)` accepts this value.
   - Selects best model by validation loss
   - Outputs:
     - `model.h5` - Trained Keras model
     - `scaler.pkl` - Input normalization

### Tab 3: Documentation

Integrated help and API reference directly in the GUI.

## Interface Walkthrough

### Dataset Generation Tab

```
📊 GENERATE DATASET
├── Upload Mechanism File
├── Configure Simulation
│   ├── Fuel Species: [CH4________]
│   └── Number of Cases: [100]
├── Temperature Range
│   ├── Min: [========800========] K
│   └── Max: [=====2300=====] K
├── Pressure Range
│   ├── Min: [=2.1=] log atm
│   └── Max: [=2.5=] log atm
├── GPS Parameters
│   ├── Alpha: [=0.001=]
│   ├── Always Threshold: [====0.99====]
│   └── Never Threshold: [=0.01=]
├── Species Ranges (JSON)
│   └── [{"CH4": [0,1], ...}]
└── [🚀 Generate Dataset] [🔄 Clear]
```

### Neural Network Tab

```
🧠 TRAIN NEURAL NETWORK
├── Input Species
│   └── [CH4, O2, CO2, H2O...]
├── Architecture
│   ├── Hidden Layers: [====2====]
│   └── Neurons per Layer: [===16===]
├── Training Settings
│   └── Learning Rate: [===0.001===]
└── [🚀 Train Neural Network]
```

## Parameters Explained

### Dataset Generation

| Parameter | Purpose | Range | Default |
|-----------|---------|-------|---------|
| **Fuel Species** | Main combustible | Any species name | CH4 |
| **N Cases** | Training simulations | 1-1000+ | 100 |
| **Temp Min/Max** | Initial temperature range | 300-3000 K | 800-2300 K |
| **Press Min/Max** | Initial pressure range | log₁₀(atm) | 2.1-2.5 |
| **Alpha** | GPS pathway threshold | 0.0001-0.1 | 0.001 |
| **Always Threshold** | "Always include" cutoff | 0.5-1.0 | 0.99 |
| **Never Threshold** | "Never include" cutoff | 0.0-0.5 | 0.01 |

### Neural Network

| Parameter | Purpose | Range | Default |
|-----------|---------|-------|---------|
| **Input Species** | ANN features | Any dataset species | CH4, O2, CO2, etc |
| **Hidden Layers** | Network depth | 1-5 | 2 |
| **Neurons** | Layer capacity | 4-256 | 16 |
| **Learning Rate** | Training speed | 0.0001-0.1 | 0.001 |
| **Number of Processes** | Parallel training workers (controls `joblib.Parallel`) | 1-64 | 1 |

## Common Workflows

### Quick Test (5 minutes)

```
1. Use default parameters, but set:
   - Number of Cases: 5 (instead of 100)
   - Temperature Range: 1000-1500 K (instead of 800-2300)
2. Upload any Cantera mechanism
3. Generate dataset
4. Train neural network with default settings
```

### Production Run

```
1. Set parameters for your application:
   - Fuel species and mechanism
   - Realistic T/P ranges
   - GPS alpha for desired accuracy
2. Set n_cases to 100-500 (more = better but slower)
3. Generate dataset (may take hours)
4. Train neural network with 2-3 hidden layers
5. Download model and scaler for deployment
```

### Fine-Tuning Architecture

To customize neural network layers beyond GUI controls:

1. Edit `src/slgps/mech_train.py`
2. Modify the `spec_train()` function
3. Save and re-train through GUI

```python
# Example: deeper network with dropout
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
```

See [API Reference](api.md) for details.

## Tips & Tricks

### Performance

- **Faster iteration**: Start with 10-20 cases instead of 100
- **Parallel training**: Ensure CPU cores available
 - **Parallel training**: Ensure CPU cores available. Use the `Number of Processes` slider in the GUI to control how many parallel workers `make_model()` launches (default in code is 28 for large runs; GUI default is 1 for safe local testing).
- **GPU training**: Install `tensorflow[and-cuda]` for 10-100x speedup
- **Storage**: Use SSD for faster I/O during data generation

### Parameters

- **Lower alpha** = more species included (more accurate, larger models)
- **Higher alpha** = fewer species (smaller, faster mechanisms)
- **More cases** = better training data (exponential time cost)
- **Deeper networks** = more model capacity (slower training)

### Troubleshooting

**Port 7860 already in use?**
```bash
python -m frontend --server_port 7861
```

**Need to share the interface?**
```bash
python -m frontend --share
```

**Check mechanism validity:**
```python
import cantera as ct
gas = ct.Solution('mech_file.cti')
print(gas.species_names)
```

## Output Files

After running the GUI, you'll have:

```
generated_data/
├── data.csv                  # State vectors (T, P, compositions)
├── species.csv              # Species importance masks
├── always_spec_nums.csv     # Always-included species
├── never_spec_nums.csv      # Never-included species
├── mechanism.cti            # Copy of uploaded mechanism
├── model.h5                 # Trained Keras ANN
└── scaler.pkl               # MinMaxScaler for normalization
```

## Next Steps

1. **Use the trained model**: See [SL_GPS.py](../src/slgps/SL_GPS.py) for adaptive simulation
2. **Visualize results**: Use `display_sim_data.py` to plot mechanism reduction
3. **Export to other tools**: Convert .h5 to .pb for OpenFOAM using `h5topb.py`

## Advanced Usage

### Share Online

```bash
python -m frontend --share
```

Creates a public link valid for 72 hours.

### Custom Output Directory

Edit `frontend/app.py` and change:
```python
data_dir = "my_output_folder"
```

### Disable Auto-Browser

```bash
python -m frontend --inbrowser false
```

## API

The GUI calls these backend functions:

- `make_data_parallel()` - Dataset generation
- `make_model()` - Neural network training

See [API Reference](api.md) for detailed documentation.

## Support

- **GUI Documentation**: See frontend/README.md
- **Full Docs**: https://ctftamu.github.io/SL-GPS/
- **GitHub**: https://github.com/ctftamu/SL-GPS
- **Issues**: https://github.com/ctftamu/SL-GPS/issues
- **Discord**: https://discord.com/channels/1333609076726431798/1333610748424880128
