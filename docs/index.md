# SL-GPS: Supervised Learning - Global Pathway Selection

Welcome to **SL-GPS**, a Python framework for automated chemistry reduction in combustion simulations. This tool uses artificial neural networks trained on chemical kinetics data to predict important species, enabling dynamic reduction of complex chemical mechanisms during autoignition simulations.

## Key Features

- **Automated Data Generation**: Run 0D autoignition simulations with adaptive GPS-based species selection
- **Neural Network Training**: Train ANNs on the generated kinetics data for species importance prediction
- **Dynamic Mechanism Reduction**: Adaptively reduce chemical mechanisms during simulation based on ANN predictions
- **Flexible Chemistry**: Support for any detailed mechanism compatible with Cantera (YAML format)
- **Parallel Processing**: Multi-process data generation and ensemble neural network training

## What This Does

SL-GPS addresses a critical challenge in combustion modeling: **large chemical mechanisms are computationally expensive, but simple ones can be inaccurate**. This framework automatically learns which species are important at different points in a simulation:

1. **Phase 1: Training Data Generation** - Runs multiple autoignition simulations with detailed mechanisms, using GPS to identify important species in each timestep interval
2. **Phase 2: ANN Training** - Trains neural networks to predict important species based on thermochemical state (temperature, pressure, composition)
3. **Phase 3: Adaptive Reduction** - Uses trained ANNs to dynamically select species during new simulations, building reduced mechanisms on-the-fly

## Typical Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Run main.py → Generate training data via GPS + ANNs      │
│    (make_data_parallel.py creates autoignition dataset)      │
├─────────────────────────────────────────────────────────────┤
│ 2. mech_train.py → Train ensemble of neural networks        │
│    (selects best model from parallel training)              │
├─────────────────────────────────────────────────────────────┤
│ 3. Run SL_GPS.py → Adaptive simulation with trained ANN     │
│    (dynamically reduces mechanism at each timestep)         │
├─────────────────────────────────────────────────────────────┤
│ 4. display_sim_data.py → Visualize results                  │
│    (temperature, HRR, species evolution, mechanism size)    │
└─────────────────────────────────────────────────────────────┘
```

## Why SL-GPS?

Traditional mechanism reduction methods (e.g., DRGEP, CSP) are slow or require manual tuning. SL-GPS provides:

- **Speed**: Once trained, ANN predictions are ~10x faster than GPS
- **Accuracy**: Learns from data-driven analysis of important reaction pathways
- **Automation**: No manual tuning of importance thresholds required
- **Adaptability**: Mechanisms reduce differently at different simulation points (e.g., pre-ignition vs. post-ignition)

## Quick Start

### 🌐 Try Online (No Installation Required)

<div style="text-align: center; margin: 30px 0;">
  <a href="https://huggingface.co/spaces/rmishra4/SL-GPS" target="_blank" style="
    display: inline-block;
    padding: 15px 40px;
    background-color: #0969da;
    color: white;
    text-decoration: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    border: 2px solid #0969da;
    transition: all 0.3s ease;
  " onmouseover="this.style.backgroundColor='#0860ca'; this.style.transform='scale(1.05)'" onmouseout="this.style.backgroundColor='#0969da'; this.style.transform='scale(1)'">
    🚀 Launch SL-GPS on Hugging Face Spaces
  </a>
</div>

**The frontend is fully deployed and ready to use in your browser.** Click the button above or [use this link](https://huggingface.co/spaces/rmishra4/SL-GPS).

### 🚀 Or Launch Locally

Alternatively, run it locally with the built-in Gradio GUI:

```bash
pip install -r frontend/requirements.txt
python -m frontend
```

This opens a browser interface where you can:
- Upload a Cantera mechanism file
- Configure simulation parameters
- Generate training datasets with one click
- Train neural networks with customizable architecture
- Download trained models

**→ [GUI Documentation](frontend.md)**

---

### Installation
```bash
pip install cantera
pip install "numpy==1.26.4" tensorflow mkdocs mkdocs-material networkx scikit-learn
```

### Manual Usage (Script-Based)
```python
# See docs/setup.md for detailed installation
# See docs/workflow.md for step-by-step tutorials
# See docs/code_structure.md for customization options
```

## Publications & References

### Primary Citation

If you use this work, please cite:
- **Mishra, R., Nelson, A., Jarrahbashi, D.**, "Adaptive global pathway selection using artificial neural networks: A-priori study", *Combustion and Flame*, **244** (2022) 112279. [[DOI: 10.1016/j.combustflame.2022.112279](https://doi.org/10.1016/j.combustflame.2022.112279)]

### Related Work

- **Gao, X., Yang, S., Sun, W.**, "A global pathway selection algorithm for the reduction of detailed chemical kinetic mechanisms", *Combustion and Flame*, **167** (2016) 238-247. [[DOI: 10.1016/j.combustflame.2016.02.007](https://doi.org/10.1016/j.combustflame.2016.02.007)]

## Contact & Community

For questions, discussions, feature requests, or bug reports:

- **GitHub Issues**: https://github.com/ctftamu/SL-GPS/issues
- **Email**:
  - Rohit Mishra: rmishra@tamu.edu
  - Aaron Nelson: aaronnelson@tamu.edu
- **Discord Community**: [Join Channel](https://discord.com/channels/1333609076726431798/1333610748424880128)

## Dependencies

Developed in **Python 3**. Core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| **Cantera** | >=3.0.0 | Chemical kinetics, mechanism handling |
| **TensorFlow** | 2.x | Neural network training (Keras API) |
| **NumPy** | 1.26.4 | Numerical computations |
| **scikit-learn** | latest | Data scaling (MinMaxScaler) |
| **NetworkX** | latest | GPS flux graph operations |
| **Pandas** | latest | Data I/O (CSV) |
| **Joblib** | latest | Parallel training |

**Note**: Cantera 3.x is required. The YAML-based mechanism format is used.

## Project Structure

```
SL-GPS/
├── docs/              # Documentation (MkDocs)
├── src/slgps/         # Main package
│   ├── main.py        # Entry point: data generation + training
│   ├── mech_train.py  # ANN training logic
│   ├── SL_GPS.py      # Adaptive simulation runner
│   ├── utils.py       # Core utilities (GPS, simulation, reduction)
│   ├── make_data_parallel.py  # Parallel data generation
│   ├── display_sim_data.py    # Results visualization
│   └── GPS/           # GPS algorithm submodule
├── tests/             # Test files and examples
├── mkdocs.yml         # Documentation configuration
└── requirements.txt   # Python dependencies
```

---

**Next Steps**:
- 📖 Read [Setup & Installation](setup.md) for detailed installation instructions
- ▶️ Check [Usage Workflow](workflow.md) for step-by-step tutorials
- 🔧 See [Code Structure](code_structure.md) for customization options