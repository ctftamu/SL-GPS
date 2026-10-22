# PODE-1 (DMM) Mechanism Validation Study

## Overview

This folder validates the SL-GPS method on the **PODE-1 mechanism** (586 species, 3091 reactions) for **dimethoxymethane** (DMM, CH₃OCH₂OCH₃, also known as PODE-1). DMM is a polyoxymethylene dimethyl ether of interest as a diesel surrogate with soot-free combustion properties.

The mechanism was developed by the **Labbé Lab** and includes comprehensive C₁–C₄ chemistry with DMM-specific pathways.

## Prerequisites

- Python 3.10+
- Cantera 3.1+
- TensorFlow 2.x, NumPy, scikit-learn, Matplotlib, Joblib
- SL-GPS installed (`pip install -e .` from repository root)
- ≥8 CPU cores recommended (parallel ensemble training)

## Running the Pipeline

### Quick start (baseline 8-case validation)
```bash
cd validation/PODEMechanism
python run_pipeline_dmm.py
```

### Full 35-case validation from scratch
```bash
python recreate_from_scratch.py
```

### Step-by-step
```bash
python run_pipeline_dmm.py    # Steps 1-5: data gen → train → validate → plot → report
python validate_full.py       # 35-case validation with trained model
python plot_final_results.py  # Generate per-case comparison plots
python plot_species_vs_time.py # Species count vs time figure
python make_final_summary.py  # Summary bar chart
```

## Mechanism Details

| Property | Value |
|----------|-------|
| Name | PODE-1 (Labbé Lab) |
| Species | 586 |
| Reactions | 3091 |
| Fuel | CH₃OCH₂OCH₃ (DMM, C₃H₈O₂, MW=76.09) |
| Format | Cantera YAML (converted from CHEMKIN) |

## Training Configuration

| Parameter | Min | Max | Distribution |
|-----------|-----|-----|--------------|
| Temperature | 800 K | 1600 K | Uniform |
| Pressure | 1 atm | 20 atm | Log-uniform |
| Equivalence ratio φ | 0.7 | 1.4 | Uniform |
| X(CH₃OCH₂OCH₃) | 0.01 | 0.05 | Uniform |
| X(O₂) | 0.10 | 0.25 | Uniform |
| X(N₂) | 0.60 | 0.80 | Uniform |
| GPS α | 0.001 | — | Fixed |

## ANN Architecture

| Parameter | Value |
|-----------|-------|
| Input features | T, P + 14 species mole fractions |
| Hidden layers | 2 × 32 neurons (ReLU) |
| Output | Sigmoid (one per variable species) |
| Loss | Binary cross-entropy |
| Ensemble | 8–12 seeds, best by validation loss |
| Early stopping | Patience 30, max 200 epochs |

**Input species:** CH₃OCH₂OCH₃, O₂, H₂O, CO₂, CO, OH, H, H₂, CH₄, CH₂O, CH₃OH, HO₂, CH₃O, HCO

## Folder Structure

```
PODEMechanism/
├── README.md
├── .gitignore
├── mechanism/
│   ├── PODE-1_Mech.inp       # Original CHEMKIN mechanism
│   ├── PODE-1_Thermo.dat     # Thermodynamic data
│   └── PODE-1.yaml           # Cantera YAML (converted)
├── data/                      # Generated training data
├── models/                    # Trained ANNs
│   └── dmm/
│       ├── model.h5
│       └── scaler.pkl
├── results/                   # Plots + simulation pickles
│   └── dmm/
├── run_pipeline_dmm.py        # Full 5-step pipeline
├── validate_full.py           # 35-case expanded validation
├── recreate_from_scratch.py   # End-to-end reproduction
├── make_balanced_data.py      # Temperature-balanced dataset
├── audit_never_list.py        # Never-list safety audit
├── plot_final_results.py      # Per-case comparison plots
├── plot_species_vs_time.py    # Species count vs time
├── make_final_summary.py      # Summary figure
└── report.tex                 # LaTeX report
```

## Notes

- The PODE-1 mechanism CHEMKIN files are converted to Cantera YAML using `ck2yaml --permissive` due to redundant thermo entries.
- Data generation with 586 species is computationally intensive (~30+ min for 15 cases).
- The `run_pipeline_dmm.py` script skips completed steps automatically.
- Delete contents of `data/`, `models/`, and `results/` to regenerate from scratch.
