# SL-GPS Copilot Instructions

## Project Overview

**SL-GPS** is a Python framework for **automated chemistry reduction** using supervised learning and the Global Pathway Selection (GPS) algorithm. It trains neural networks (ANNs) on chemical kinetics data to predict which species are important for accurate mechanism reduction during autoignition simulations.

### Core Workflow
1. **Data Generation**: Run `main.py` to generate autoignition simulation data using detailed chemical mechanisms (Cantera)
2. **ANN Training**: Train neural networks (via TensorFlow/Keras) to predict species inclusion
3. **Adaptive Simulation**: Run `SL_GPS.py` to test ANNs on new simulations, dynamically reducing chemical mechanisms

## Critical Architecture

### Three-Phase Process (main.py → mech_train.py → SL_GPS.py)

**Phase 1: Training Data Generation** (`main.py` calls `make_data_parallel()`)
- Runs **parallel autoignition simulations** across randomized initial conditions (T, P, composition)
- Uses **Global Pathway Selection (GPS)** algorithm (submodule at `src/slgps/GPS/`) to extract important species from each simulation interval
- Adaptive threshold: splits simulation into ignition (fast timesteps) and post-ignition phases (coarser timesteps)
- Outputs: `data_path/data.csv` (state vectors) and `data_path/species.csv` (binary species masks)

**Phase 2: Neural Network Training** (`mech_train.py::make_model()`)
- Loads state vectors and species data from training directory
- Normalizes input using MinMaxScaler (saved as `.pkl` for later use)
- **Trains ensemble of ANNs in parallel** (default: 28 processes) with train/validation split
- Architecture customization: Edit `spec_train()` function to modify hidden layers (default: 16 neurons, relu activation)
- Selects best model by validation loss, saves as `.h5` file

**Phase 3: Adaptive Simulation** (`SL_GPS.py` calls `utils.py::auto_ign_build_SL()`)
- Runs new autoignition simulation with detailed mechanism
- At each timestep, calls ANN to predict which species to include
- Dynamically builds reduced mechanisms via `utils.py::sub_mech()` (filters reactions to only use predicted species)
- Species partitioned into three groups: **variable** (ANN-predicted), **always** (threshold > 99%), **never** (threshold < 1%)
- Outputs simulation results (T, HRR, composition, reaction counts) as pickle file

### GPS Integration

The embedded GPS module (`src/slgps/GPS/`) handles flux analysis:
- `core/def_GPS.py`: Graph-based pathway selection algorithm
- `ct/def_ct_tools.py`: Cantera solution utilities
- Called within `utils.py::GPS_spec()` to identify important species for each simulation interval
- Configured with: `fuel` (e.g., 'CH4'), `alpha` (pathway threshold, default 0.001), `sources`/`targets` (fuel/products)

## Key Files & Patterns

| File | Purpose | Key Parameters |
|------|---------|-----------------|
| `main.py` | Entry point for training | `n_cases`, `t_rng`, `p_rng`, `alpha`, `always_threshold`, `never_threshold` |
| `mech_train.py` | ANN architecture & training | `spec_train()` function contains layers (edit here to customize); `num_processes` is configurable (default 28) and exposed in the Gradio frontend |
| `SL_GPS.py` | Simulation runner | `norm_Dt`, `ign_Dt` (timestep control), `T0_in`, `phi`, `atm` (initial conditions) |
| `utils.py` | Core simulation logic | `auto_ign_build_SL()` (main loop), `GPS_spec()` (species selection), `sub_mech()` (mechanism building) |
| `make_data_parallel.py` | Data generation | `process_simulation()` (parallel job), coordinate ignition detection |
| `display_sim_data.py` | Result visualization | Matplotlib plotting of `.pkl` simulation results |

## Project-Specific Conventions

1. **Mechanism Files**: Use Cantera CTI format (`.cti`). Examples: `gri30.cti`, `nHeptane.cti`, or custom paths
2. **Output Paths**: No intermediate cleanup; directory creation is handled by `os.makedirs(exist_ok=True)`
3. **Parallel Execution**: Uses `joblib.Parallel` for ANN training (default 28 processes). `num_processes` is configurable via `make_model(..., num_processes=...)` and the Gradio frontend exposes a slider to control it.
4. **Model Format**: ANNs saved as Keras `.h5` files; scalers as joblib pickle files
5. **State Vectors**: CSV files with columns `[# Temperature, Atmospheres, <species mole fractions>]`
6. **Species Naming**: Must match Cantera species names exactly (case-sensitive, e.g., 'CH4', 'H2O', 'OH')

## Common Workflows

**Generate new training data:**
```python
# In main.py, set n_cases, temperature/pressure ranges, species_ranges, alpha
# Leave data_path to non-existent directory to trigger generation
python src/slgps/main.py  # Runs make_data_parallel then make_model
```

**Customize ANN architecture:**
```python
# Edit mech_train.py::spec_train() - add/remove Dense layers before output layer
# Default: 16 neurons, relu; output: sigmoid (binary classification per species)
```

**Run adaptive simulation with trained ANN:**
```python
# In SL_GPS.py, set paths to trained model/scaler, initial conditions, timesteps
python src/slgps/SL_GPS.py  # Outputs results_path as .pkl
```

**Visualize results:**
```python
# Edit display_sim_data.py with path to .pkl file
python src/slgps/display_sim_data.py  # Shows T, HRR, species, mechanism size vs time
```

## Dependencies & Version Constraints

- **Cantera 2.6.0** (exact): Chemical kinetics; core dependency for mechanism handling
- **TensorFlow 2.x**: Keras API for neural networks
- **NumPy 1.26.4**: Numerical computations
- **scikit-learn**: MinMaxScaler for input normalization
- **NetworkX**: GPS flux graph operations
- **Pandas**: Data I/O (CSV)
- **Joblib**: Parallel training in mech_train.py

**Critical:** Cantera version must be exactly 2.6.0; newer versions have breaking API changes.

## Testing

Test files in `tests/` are minimal (only import checks). Primary validation occurs via:
1. Running `main.py` end-to-end (data generation → training)
2. Running `SL_GPS.py` with trained model (adaptive simulation)
3. Visual inspection of results via `display_sim_data.py`

Jupyter notebook `tests/converth5ToPb.ipynb` demonstrates model export to frozen graph format (for external tools like OpenFOAM).

## Important Caveats

- **No graceful rollback**: If data generation fails mid-way, the incomplete directory remains and won't retrigger generation on rerun
- **Parallel training flakiness**: Occasionally, best_model selection may vary due to random initialization; consider manual averaging of results
- **Species order matters**: `input_specs` in `main.py` and `SL_GPS.py` must align for consistent results
- **Always/Never thresholds**: Species with 99%+ occurrence (always) or <1% occurrence (never) are excluded from ANN training; adjust if too few/many variable species

## Development Notes

- GPS algorithm is computationally expensive; reduce `GPS_per_interval` or `n_cases` for faster iteration
- Mechanism files must have reactions defined for all input species; missing reactions cause simulation crashes
- Early stopping patience in `mech_train.py` is 30 epochs; adjust if training oscillates
