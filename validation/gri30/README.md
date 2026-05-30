# GRI-Mech 3.0 Validation Study

## Overview

This folder validates the SL-GPS method on **GRI-Mech 3.0** (53 species, 325 reactions) for CH₄/air autoignition. It serves as a fast baseline demonstration of the full pipeline.

## Prerequisites

- Python 3.10+
- Cantera 2.6.0 (GRI-Mech 3.0 is built-in)
- TensorFlow 2.x, NumPy 1.26.4, scikit-learn, Matplotlib, Joblib
- SL-GPS installed (`pip install -e .` from repository root)

## Running the Pipeline

```bash
cd validation/gri30
python run_pipeline.py
```

This single script executes the full workflow end-to-end (~2 minutes):

### Step 1: Training Data Generation (GPS)

Runs 20 parallel autoignition simulations with randomized initial conditions and applies GPS (α = 0.001) at each timestep interval to label important species.

| Parameter | Range |
|-----------|-------|
| Temperature | 900–2000 K (uniform) |
| Pressure | 1–20 atm (log-uniform) |
| CH₄ | 0.02–0.10 |
| O₂ | 0.10–0.25 |
| N₂ | 0.60–0.80 |
| CO₂ | 0.00–0.005 |
| H₂O | 0.00–0.03 |

**Output:** `data/train/data.csv` (state vectors), `data/train/species.csv` (binary species masks)

### Step 2: Neural Network Training

Trains an ensemble of 4 ANNs in parallel and selects the best by validation loss.

| Parameter | Value |
|-----------|-------|
| Input features | T, P + 10 species (CH₄, H₂O, OH, H, CO, O₂, CO₂, O, CH₃, H₂) |
| Architecture | 1 hidden layer × 16 neurons (ReLU) |
| Output | Sigmoid (one per variable species) |
| Loss | Binary cross-entropy |
| Early stopping | Patience 30, max 200 epochs |

**Output:** `models/gri30_model.h5`, `models/gri30_scaler.pkl`

### Step 3: Validation Simulations

Runs 6 unseen autoignition cases comparing the detailed mechanism against SL-GPS:

| Case | T (K) | P (atm) | φ | Description |
|------|--------|----------|---|-------------|
| T1000_P1 | 1000 | 1 | 1.0 | Low-T, low-P |
| T1000_P10 | 1000 | 10 | 1.0 | Low-T, high-P |
| T1500_P1 | 1500 | 1 | 1.0 | Mid-T, low-P |
| T1500_P10 | 1500 | 10 | 1.0 | Mid-T, high-P |
| T2000_P1 | 2000 | 1 | 1.0 | High-T, low-P |
| T1200_lean | 1200 | 5 | 0.6 | Lean mixture |

**Output:** `results/<case>.pkl` (pickle with detailed + SL-GPS simulation data)

### Step 4: Plot Generation

Generates comparison figures (temperature, HRR, species profiles, mechanism size) plus summary plots for ignition delay and reduction ratio.

**Output:** `results/` (PNG figures), `report.tex` and `report.pdf` (LaTeX report at root)

## Folder Structure

```
gri30/
├── README.md          # This file
├── report.tex         # LaTeX validation report
├── report.pdf         # Compiled report
├── mechanism/         # GRI-Mech 3.0 is built into Cantera (placeholder)
├── data/              # Generated training data (not tracked, regenerate with pipeline)
├── models/            # Trained ANN (.h5) and scaler (.pkl)
├── results/           # Plots (tracked) + simulation pickles (regenerate with pipeline)
└── run_pipeline.py    # Full pipeline script
```

## Notes

- GRI-Mech 3.0 is built into Cantera — no external mechanism file needed (the `mechanism/` folder is a placeholder for consistency).
- The full pipeline runs in ~2 minutes on a modern machine.
- To regenerate from scratch, delete the `data/`, `models/`, and `results/` contents and re-run.
- Training data and simulation pickles are not tracked in git; run the pipeline to regenerate them.
