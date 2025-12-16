---
title: SL-GPS Chemistry Reduction GUI
emoji: ⚗️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.29.0"
app_file: app.py
pinned: false
license: mit
---

# SL-GPS Chemistry Reduction GUI

This is a Hugging Face Spaces deployment of the **SL-GPS** (Supervised Learning - Global Pathway Selection) framework for automated chemistry reduction in combustion simulations.

## 🎯 About SL-GPS

SL-GPS is a Python framework that uses artificial neural networks trained on chemical kinetics data to predict which species are important during combustion simulations, enabling **dynamic reduction of complex chemical mechanisms**.

### Key Features

- 🧪 **Upload Cantera Mechanisms** - Support for any CTI format mechanism
- ⚙️ **Configure Simulations** - Set temperature, pressure, and composition ranges
- 🧠 **Train Neural Networks** - Customize ANN architecture with easy sliders
- 📊 **Real-time Progress** - Monitor dataset generation and training
- 💾 **Download Results** - Export trained models and scalers

## 📖 How to Use

### Tab 1: Generate Dataset

1. Upload your Cantera mechanism file (`.cti` format)
2. Configure parameters:
   - **Fuel Species** - e.g., CH4, H2, or nHeptane
   - **Number of Simulations** - More simulations = better training data (try 5-10 for testing, 100+ for production)
   - **Temperature Range** - Initial temperature in Kelvin (e.g., 800-2300 K)
   - **Pressure Range** - Log₁₀ of pressure in atmospheres
   - **GPS Alpha** - Pathway importance threshold (lower = more species included)
   - **Species Composition Ranges** - Min/max mole fractions as JSON
3. Click **Generate Dataset** and wait for completion

**Output:**
- `data.csv` - State vectors (temperature, pressure, species)
- `species.csv` - Binary species importance masks

### Tab 2: Train Neural Network

1. Specify **Input Species** - Which species the ANN uses as features
2. Configure architecture:
   - **Hidden Layers** - Network depth (1-5)
   - **Neurons per Layer** - Capacity (4-256)
   - **Parallel Workers** - Number of parallel training processes
   - **Learning Rate** - Training speed
3. Click **Train Neural Network** and monitor progress

**Output:**
- `model.h5` - Trained Keras neural network
- `scaler.pkl` - Input normalization scaler

### Tab 3: Documentation

Access built-in help and API reference for advanced users.

## 📚 Full Documentation

For detailed tutorials, API reference, and advanced usage:

**→ [Full Documentation](https://ctftamu.github.io/SL-GPS/)** (Official docs)

**→ [GitHub Repository](https://github.com/ctftamu/SL-GPS)** (Original repository)

## 🔧 Technical Details

### Requirements

- **Python 3.10+**
- **Cantera 2.6.0** (exact version required)
- **TensorFlow 2.x**
- **Gradio 4.0+**
- **scikit-learn**, **NetworkX**, **Pandas**, **NumPy**

### Computing Time

- **Dataset Generation**: 5-30 minutes depending on number of simulations
- **Neural Network Training**: 5-15 minutes depending on dataset size and parallel workers
- **Total Time**: ~30-45 minutes for a full workflow

### Mechanism Compatibility

This tool works with any Cantera mechanism in CTI format, including:
- GRI-Mech 3.0 (natural gas)
- nHeptane mechanisms
- Custom mechanisms (must have proper species and reaction definitions)

## 📊 Typical Workflow

```
1. Upload mechanism file
   ↓
2. Configure simulation parameters
   ↓
3. Generate training dataset (GPS analysis)
   ↓
4. Configure ANN architecture
   ↓
5. Train neural network on generated data
   ↓
6. Download model.h5 and scaler.pkl
   ↓
7. Use in production with SL_GPS.py for adaptive simulations
```

## 📄 Citation

If you use this tool, please cite:

> Mishra, R., Nelson, A., Jarrahbashi, D., "Adaptive global pathway selection using artificial neural networks: A-priori study", *Combustion and Flame*, **244** (2022) 112279.
>
> DOI: [10.1016/j.combustflame.2022.112279](https://doi.org/10.1016/j.combustflame.2022.112279)

## 📞 Support

- **Issues & Questions**: [GitHub Issues](https://github.com/ctftamu/SL-GPS/issues)
- **Email**: rmishra@tamu.edu
- **Discord**: [Join Community](https://discord.com/channels/1333609076726431798/1333610748424880128)

## ⚖️ License

MIT License - See repository for details.

---

**Last Updated**: December 2024  
**Maintained by**: Texas A&M Reactive Flow Research Lab
