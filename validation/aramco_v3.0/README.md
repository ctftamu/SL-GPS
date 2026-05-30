# AramcoMech 3.0 Validation Study

## Overview

This folder validates the SL-GPS method on **AramcoMech 3.0** (581 species, 3037 reactions) for three fuels: methane (CH₄), propane (C₃H₈), and ethylene (C₂H₄). This demonstrates SL-GPS scalability to large, industrially relevant mechanisms.

## Prerequisites

- Python 3.10+
- Cantera 2.6.0
- TensorFlow 2.x, NumPy 1.26.4, scikit-learn, Matplotlib, Joblib
- SL-GPS installed (`pip install -e .` from repository root)
- ≥8 CPU cores recommended (parallel ensemble training)

## Obtaining the Mechanism

AramcoMech 3.0 must be downloaded from:
https://www.universityofgalway.ie/combustionchemistrycentre/mechanismdownloads/

1. Download the CHEMKIN-format mechanism and thermodynamic data
2. Convert to Cantera CTI format using:
   ```bash
   ck2cti --input=chem.inp --thermo=therm.dat --transport=tran.dat --output=aramco_v3.cti
   ```
3. Place the resulting `aramco_v3.cti` file in the `mechanism/` directory

## Running the Pipeline

Each fuel has its own self-contained pipeline script:

```bash
cd validation/aramco_v3.0

# Methane (fastest, ~2 min)
python run_pipeline.py

# Propane (~30 min)
python run_pipeline_c3h8.py

# Ethylene (~34 min)
python run_pipeline_c2h4.py
```

Alternatively, run individual steps:
```bash
python run_all.py            # Full orchestrated pipeline (CH4)
python run_all.py --from 3   # Resume from step 3
```

Or step-by-step:
```bash
python 01_generate_data.py
python 02_train_model.py
python 03_validate.py
python 04_plot_results.py
```

### Step 1: Training Data Generation (GPS)

Runs serial autoignition simulations (10–30 cases depending on fuel) with randomized initial conditions. At each timestep interval, GPS (α = 0.001) identifies important species.

**Note:** Data generation is the computational bottleneck for large mechanisms (~23 min for 15 cases with 581 species).

**Output:** `data/train/data.csv`, `data/train/species.csv`, threshold CSVs

### Step 2: Neural Network Training

Trains an ensemble of 8 ANNs in parallel, selects the best by validation loss.

| Parameter | Value |
|-----------|-------|
| Input features | T, P + 14 key species mole fractions |
| Architecture | 2 hidden layers × 24–32 neurons (ReLU) |
| Output | Sigmoid (one per variable species) |
| Loss | Binary cross-entropy |
| Early stopping | Patience 30, max 200 epochs |

**Output:** `models/<fuel>/` (model.h5 + scaler.pkl per fuel)

### Step 3: Validation Simulations

Runs 4–8 unseen autoignition cases comparing detailed mechanism vs SL-GPS.

### Step 4: Plot Generation & Report

Generates comparison figures and compiles a LaTeX report (`report.pdf` at root).

## Training Conditions

| Parameter | Min | Max | Distribution |
|-----------|-----|-----|--------------|
| Temperature | 900 K | 2000 K | Uniform |
| Pressure | 1 atm | 20 atm | Log-uniform |
| CH₄ | 0.02 | 0.10 | Uniform |
| O₂ | 0.10 | 0.25 | Uniform |
| N₂ | 0.60 | 0.80 | Uniform |
| CO₂ | 0.00 | 0.005 | Uniform |
| H₂O | 0.00 | 0.03 | Uniform |

## Validation Points (CH₄)

| Case | T (K) | P (atm) | φ | Description |
|------|--------|----------|---|-------------|
| T1000_P1 | 1000 | 1 | 1.0 | Low-T, low-P |
| T1500_P1 | 1500 | 1 | 1.0 | Mid-T, low-P |
| T1500_P10 | 1500 | 10 | 1.0 | Mid-T, high-P |
| T2000_P1 | 2000 | 1 | 1.0 | High-T, low-P |

## Folder Structure

```
aramco_v3.0/
├── README.md              # This file
├── report.tex             # LaTeX validation report
├── report.pdf             # Compiled report
├── mechanism/             # Mechanism files (.cti + raw CHEMKIN)
│   ├── aramco_v3.cti
│   ├── ARAMCO3.0.MECH
│   ├── ARAMCO3.0.THERM
│   └── ARAMCO3.0.TRAN
├── data/                  # Generated training data (not tracked)
├── models/                # Trained ANNs (tracked)
│   ├── ch4/
│   │   ├── model.h5
│   │   └── scaler.pkl
│   ├── c3h8/
│   │   ├── model.h5
│   │   └── scaler.pkl
│   └── c2h4/
│       ├── model.h5
│       └── scaler.pkl
├── results/               # Plots (tracked) + simulation pickles (not tracked)
│   ├── ch4/
│   ├── c3h8/
│   └── c2h4/
├── run_pipeline.py        # CH4 full pipeline
├── run_pipeline_c3h8.py
├── run_pipeline_c2h4.py
├── 01_generate_data.py    # Step-by-step scripts (CH4)
├── 02_train_model.py
├── 03_validate.py
├── 04_plot_results.py
└── run_all.py             # Orchestration script
```

## Notes

- AramcoMech 3.0 is NOT built into Cantera — you must obtain and convert it (see above).
- GPS data generation with 581 species is significantly slower than GRI-Mech 3.0.
- To regenerate from scratch, delete contents of `data/`, `models/`, and `results/` and re-run.
- The `run_pipeline*.py` scripts skip completed steps automatically (delete outputs to re-run).
- Training data and simulation pickles are not tracked in git; run the pipeline to regenerate them.
