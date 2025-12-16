# SL-GPS Frontend - Quick Reference

## 🌐 Try Online First

**[Launch on Hugging Face Spaces](https://huggingface.co/spaces/ctftamu/SL-GPS)** - No installation required!

<iframe
  src="https://huggingface.co/spaces/ctftamu/SL-GPS/embed"
  frameborder="0"
  width="100%"
  height="1200"
  style="border-radius: 8px; margin: 20px 0;"
></iframe>

---

## Installation & Launch Locally

```bash
# Install frontend dependencies
pip install -r frontend/requirements.txt

# Launch the GUI
python -m frontend

# Alternative methods
python frontend/app.py
gradio frontend/app.py
```

**Browser opens automatically at:** `http://localhost:7860`

---

## Frontend Features

### 📊 Tab 1: Generate Dataset

Upload a Cantera mechanism file and configure:

- **Fuel Species** - e.g., CH4, H2
- **Number of Simulations** - Training cases (more = better but slower)
- **Temperature Range** - Initial T (K)
- **Pressure Range** - Initial P (log₁₀ atm)
- **GPS Alpha** - Pathway importance threshold
- **Always/Never Thresholds** - Species frequency cutoffs
- **Species Composition Ranges** - Min/max mole fractions (JSON)

**Output:** `generated_data/` folder with:
- `data.csv` - State vectors
- `species.csv` - Species masks
- `model.h5` - Trained ANN (after training)
- `scaler.pkl` - Input normalization

### 🧠 Tab 2: Train Neural Network

Configure and train:

- **Input Species** - Which species the ANN uses as features
- **Hidden Layers** - Network depth (1-5)
- **Neurons per Layer** - Capacity (4-256)
- **Learning Rate** - Training speed

**Note:** GUI shows layer/neuron controls, but to apply custom architectures, edit `src/slgps/mech_train.py::spec_train()`

### 📖 Tab 3: Documentation

Built-in help and API reference.

---

## File Structure

```
frontend/
├── __init__.py          # Package initialization
├── __main__.py          # Launch script (python -m frontend)
├── app.py               # Main Gradio application (~450 lines)
├── requirements.txt     # Dependencies (gradio>=4.0.0)
└── README.md            # Detailed frontend documentation
```

---

## Typical Workflow

```
1. Launch: python -m frontend
2. Upload mechanism file (.cti)
3. Set parameters (fuel, n_cases, T/P ranges, GPS alpha)
4. Click "Generate Dataset" → Wait for completion
5. Switch to "Train Neural Network" tab
6. Configure NN architecture
7. Click "Train" → Wait for training
8. Download model.h5 and scaler.pkl
9. Use in SL_GPS.py for adaptive simulations
```

---

## Parameters Guide

### Dataset Generation

| Parameter | Purpose | Default | Tips |
|-----------|---------|---------|------|
| Fuel | Combustible species | CH4 | Must exist in mechanism |
| N Cases | Simulations | 100 | Use 5-10 for testing, 100+ for production |
| Temp Min/Max | Temperature range (K) | 800-2300 | Match your application |
| Press Min/Max | Pressure range (log atm) | 2.1-2.5 | log₁₀(pressure in atm) |
| Alpha | GPS threshold | 0.001 | Lower = more species included |
| Always | Species frequency cutoff | 0.99 | Species in >99% of cases |
| Never | Species frequency cutoff | 0.01 | Species in <1% of cases |

### Neural Network

| Parameter | Purpose | Default | Tips |
|-----------|---------|---------|------|
| Input Species | ANN features | CH4, O2, CO2, etc | Should be most important species |
| Layers | Network depth | 2 | More = more complex patterns |
| Neurons | Layer size | 16 | More = larger capacity |
| Learning Rate | Training speed | 0.001 | Smaller = slower but more stable |

---

## Common Issues & Solutions

### "Port 7860 already in use"
```bash
python -m frontend --server_port 7861
```

### "SL-GPS not installed" warning
```bash
pip install -e .
```

### Mechanism file not uploading
- Ensure it's in `.cti` format
- Check file is valid Cantera mechanism
- Verify all species exist in mechanism

### Dataset generation slow
- Reduce `n_cases` (try 10-20)
- Reduce `GPS_per_interval` in code
- Use SSD storage for faster I/O

### Training takes forever
- Reduce input species count
- Reduce `n_cases` if using small data
- Install tensorflow[and-cuda] for GPU

---

## Advanced Features

### Share Online
```bash
python -m frontend --share
```
Creates public link (valid 72 hours).

### Disable Auto-Browser
```bash
python -m frontend --inbrowser false
```

### Custom Output Directory
Edit `frontend/app.py`:
```python
data_dir = "my_custom_folder"
```

---

## Output Files Explained

### data.csv
```
# Temperature,Atmospheres,CH4,O2,N2,...
1500,1.0,0.05,0.21,0.74,...
```
State vectors with T, P, and species mole fractions.

### species.csv
```
CH4,O2,N2,CO2,H2O,...
1,1,1,0,0,...
```
Binary masks: 1=important species, 0=not important.

### model.h5
Trained Keras neural network in HDF5 format.

### scaler.pkl
MinMaxScaler (0-1 normalization) for input preprocessing.

---

## Next Steps After Training

1. **Use in Simulations**: Load model/scaler in `SL_GPS.py`
2. **Visualize**: Use `display_sim_data.py` to plot results
3. **Export**: Convert to `.pb` with `h5topb.py` for OpenFOAM
4. **Share**: Upload to model repository

---

## Architecture Customization

To modify the neural network beyond GUI controls:

1. Edit `src/slgps/mech_train.py`
2. Find `spec_train()` function
3. Add layers before output:

```python
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
# Output layer follows automatically
```

See [API Reference](api.md) for full details.

---

## Dependencies

- **gradio** (4.0+) - Web UI framework
- **slgps** - Main package
- **tensorflow**, **cantera**, **numpy**, **scikit-learn** - Backend

Install all with:
```bash
pip install -r requirements.txt
pip install -r frontend/requirements.txt
```

---

## Support & Help

- **Frontend Docs**: `frontend/README.md`
- **Full Docs**: https://ctftamu.github.io/SL-GPS/
- **GitHub**: https://github.com/ctftamu/SL-GPS
- **Issues**: https://github.com/ctftamu/SL-GPS/issues
- **Email**: rmishra@tamu.edu

---

**Version:** 1.0.0  
**Last Updated:** December 2024
