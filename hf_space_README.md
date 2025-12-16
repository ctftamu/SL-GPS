# SL-GPS Chemistry Reduction GUI

This is a Hugging Face Spaces deployment of the SL-GPS (Supervised Learning - Global Pathway Selection) chemistry reduction GUI.

## About SL-GPS

SL-GPS is a Python framework for automated chemistry reduction in combustion simulations. This Gradio interface allows you to:

1. **Generate Training Datasets** - Upload a Cantera mechanism and run autoignition simulations with GPS-based species selection
2. **Train Neural Networks** - Train ANNs to predict important species with customizable architecture
3. **Download Models** - Save trained models and scalers for use in adaptive simulations

## Features

- 🧪 Upload any Cantera mechanism (CTI format)
- ⚙️ Configure simulation parameters (temperature, pressure, composition ranges)
- 🧠 Customize neural network architecture (hidden layers, neurons, parallelism)
- 📊 Real-time training progress and status updates
- 💾 Download trained models and scalers

## How to Use

1. **Generate Dataset Tab**
   - Upload your Cantera mechanism file
   - Set simulation parameters (fuel species, temperature range, etc.)
   - Configure GPS parameters (alpha, thresholds)
   - Click "Generate Dataset"

2. **Train Neural Network Tab**
   - Specify input species for the ANN
   - Configure network architecture (hidden layers, neurons per layer)
   - Set number of parallel training workers
   - Click "Train Neural Network"

3. **Documentation Tab**
   - Learn more about SL-GPS and parameters

## Requirements

This Space runs with:
- Python 3.10
- TensorFlow 2.15+
- Cantera 2.6.0
- Gradio 4.0+

## Links

- **Full Documentation**: https://ctftamu.github.io/SL-GPS/
- **GitHub Repository**: https://github.com/ctftamu/SL-GPS
- **Paper**: [Mishra et al., 2022](https://doi.org/10.1016/j.combustflame.2022.112279)

## Notes

- Dataset generation can take several minutes depending on number of simulations
- Training time depends on dataset size and number of parallel workers
- For best results, use mechanisms with 50-200 species
- Downloaded models can be used with the main SL-GPS Python package for adaptive simulations

## Citation

If you use this tool, please cite:

Mishra, R., Nelson, A., Jarrahbashi, D., "Adaptive global pathway selection using artificial neural networks: A-priori study", *Combustion and Flame*, **244** (2022) 112279.
