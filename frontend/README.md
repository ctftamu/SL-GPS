# SL-GPS Frontend

A beautiful, browser-based GUI for the SL-GPS chemistry reduction framework using Gradio.

## Features

- 🎨 **Intuitive Interface**: Easy-to-use tabs for dataset generation and NN training
- 📁 **File Upload**: Upload Cantera mechanism files directly
- ⚙️ **Parameter Control**: Configure all simulation and training parameters via sliders and input fields
- 🧠 **Neural Network Customization**: Control number of hidden layers and neurons
- 📊 **Real-time Status**: Monitor progress with status messages
- 🌐 **Browser-Based**: No installation needed beyond Python package
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile

## Installation

### Prerequisites
- Python 3.8+
- SL-GPS installed (`pip install git+https://github.com/ctftamu/SL-GPS.git`)

### Quick Install

```bash
# Clone repository
git clone https://github.com/ctftamu/SL-GPS.git
cd SL-GPS

# Install main dependencies
pip install -r requirements.txt

# Install frontend dependencies
pip install -r frontend/requirements.txt
```

## Usage

### Method 1: Launch with Python Module (Recommended)

```bash
# From SL-GPS root directory
python -m frontend
```

The GUI will open automatically in your default browser at `http://localhost:7860`

### Method 2: Direct Script Execution

```bash
python frontend/app.py
```

### Method 3: Using Gradio CLI

```bash
gradio frontend/app.py
```

## Workflow

### Step 1: Generate Dataset

1. Click the **📊 Generate Dataset** tab
2. Upload a Cantera mechanism file (.cti)
3. Configure parameters:
   - Fuel species (e.g., CH4, H2)
   - Number of simulations
   - Temperature and pressure ranges
   - GPS parameters (alpha, thresholds)
   - Species composition ranges
4. Click **🚀 Generate Dataset**
5. Monitor progress in the status box

**Output**: 
- `generated_data/data.csv` - State vectors
- `generated_data/species.csv` - Species importance masks
- `generated_data/always_spec_nums.csv` - Always-included species
- `generated_data/never_spec_nums.csv` - Never-included species

### Step 2: Train Neural Network

1. Click the **🧠 Train Neural Network** tab
2. Specify input species (comma-separated)
3. Configure architecture:
   - Number of hidden layers
   - Neurons per layer
   - Learning rate
4. Click **🚀 Train Neural Network**
5. Monitor training progress

**Output**:
- `generated_data/model.h5` - Trained Keras model
- `generated_data/scaler.pkl` - Input normalization scaler

## Parameters Explained

### Dataset Generation

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **Fuel Species** | Main reactant species | CH4 | Any valid species |
| **Number of Simulations** | Autoignition cases to run | 100 | 1+ |
| **Temperature Range** | Initial T in Kelvin | 800-2300 K | 300-3000 K |
| **Pressure Range** | log₁₀(Pressure in atm) | 2.1-2.5 | 0-5 |
| **GPS Alpha** | Pathway importance threshold | 0.001 | 0.0001-0.1 |
| **Always Threshold** | Species occurrence for "always include" | 0.99 | 0.5-1.0 |
| **Never Threshold** | Species occurrence for "never include" | 0.01 | 0.0-0.5 |

### Neural Network

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **Input Species** | Features fed to ANN | CH4, O2, CO2, etc | Any species in dataset |
| **Hidden Layers** | Number of hidden layers | 2 | 1-5 |
| **Neurons per Layer** | Nodes in each hidden layer | 16 | 4-256 |
| **Learning Rate** | Training step size | 0.001 | 0.0001-0.1 |

## Customizing Neural Network Architecture

The GUI shows layer/neuron controls, but to apply custom architectures, you need to edit the backend:

1. Open `src/slgps/mech_train.py`
2. Find the `spec_train()` function
3. Modify the Dense layers before the output layer:

```python
def spec_train(X_train, Y_train):
    model = tf.keras.Sequential()
    
    # Add your custom layers here
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'))
    
    # Output layer (don't modify)
    model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))
```

See the [API Reference](https://ctftamu.github.io/SL-GPS/api/) for more details.

## Troubleshooting

### Issue: Port 7860 already in use

```bash
# Use a different port
python -m frontend --server_port 7861
```

### Issue: "SL-GPS not installed" warning

```bash
# Ensure SL-GPS is installed
pip install -e .
```

### Issue: Mechanism file not recognized

- Ensure the file is in Cantera CTI format (.cti)
- Check that all species in your configuration exist in the mechanism
- Verify mechanism file syntax using Cantera directly

### Issue: Dataset generation fails

- Ensure mechanism file has valid reactions
- Check that fuel species exists in mechanism
- Verify composition ranges don't exceed [0, 1]
- Check available disk space for output files

## Advanced Usage

### Share Your Interface Online

```bash
python -m frontend --share
```

This creates a public link (valid for 72 hours) that others can access without running locally.

### Change Output Directory

Edit `app.py` and change:
```python
data_dir = "custom_output_path"
```

### Disable Browser Launch

```bash
python -m frontend --server_port 7860 --inbrowser false
```

## Architecture

```
frontend/
├── __init__.py          # Package init
├── __main__.py          # Entry point for 'python -m frontend'
├── app.py               # Main Gradio application
├── requirements.txt     # Frontend dependencies
└── README.md            # This file
```

## Dependencies

- **gradio** (4.0+) - Web UI framework
- **slgps** - Chemistry reduction backend

## File Formats

### Input: Mechanism File (.cti)

Cantera chemical kinetics file format. Examples:
- `gri30.cti` - GRI-Mech 3.0 mechanism
- `nHeptane.cti` - Heptane combustion mechanism
- Custom mechanisms in CTI format

### Output: Generated Files

**data.csv** (state vectors):
```
# Temperature,Atmospheres,CH4,O2,N2,...
1500,1.0,0.05,0.21,0.74,...
1550,1.0,0.04,0.20,0.76,...
```

**species.csv** (importance masks):
```
CH4,O2,N2,CO2,H2O,...
1,1,1,0,0,...
1,1,1,1,1,...
```

**model.h5**: Keras neural network (binary format)
**scaler.pkl**: MinMaxScaler for input normalization (pickle format)

## Performance Tips

- **Faster iteration**: Reduce `n_cases` for testing (e.g., 10-20 instead of 100)
- **Parallel training**: Ensure CPU cores available (default uses 28 processes)
- **GPU acceleration**: Install tensorflow with CUDA for 10-100x speedup
- **Large simulations**: Use SSD storage for faster I/O

## API Documentation

Full API reference: https://ctftamu.github.io/SL-GPS/api/

Key functions:
- `make_data_parallel()` - Generate training data
- `make_model()` - Train neural network
- `auto_ign_build_SL()` - Run adaptive simulation

## Citation

If you use this GUI in your work, please cite:

```bibtex
@article{mishra2022adaptive,
  title={Adaptive global pathway selection using artificial neural networks: A-priori study},
  author={Mishra, Rohit and Nelson, Aaron and Jarrahbashi, Dariush},
  journal={Combustion and Flame},
  volume={244},
  pages={112279},
  year={2022}
}
```

## License

Same as SL-GPS main repository (see LICENSE file)

## Support

- **Documentation**: https://ctftamu.github.io/SL-GPS/
- **GitHub Issues**: https://github.com/ctftamu/SL-GPS/issues
- **Discord**: https://discord.com/channels/1333609076726431798/1333610748424880128
- **Email**: rmishra@tamu.edu, aaronnelson@tamu.edu

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Version**: 1.0.0  
**Last Updated**: December 2024
