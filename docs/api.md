# API Reference

This page documents the key functions and modules in SL-GPS.

## Core Modules

### `main.py` - Training Pipeline Entry Point

Main orchestration script that coordinates data generation and ANN training.

**Key Variables** (edit these to customize your run):

```python
fuel = 'CH4'  # Fuel species name
mech_file = 'gri30.cti'  # Detailed mechanism file (Cantera CTI format)
n_cases = 100  # Number of training simulations
t_rng = [800, 2300]  # Temperature range (K)
p_rng = [2.1, 2.5]  # Pressure range (log atm)
alpha = 0.001  # GPS error tolerance (smaller = more species)
always_threshold = 0.99  # Species in >99% of cases (always included)
never_threshold = 0.01  # Species in <1% of cases (never included)
data_path = 'TrainingData/MyData'  # Where to save generated data
model_path = 'Models/MyModel.h5'  # Where to save trained ANN
scaler_path = 'Scalers/MyScaler.pkl'  # Where to save input scaler
```

**Workflow**:
```python
if __name__ == '__main__':
    # If data doesn't exist, generate it
    if not os.path.exists(data_path):
        make_data_parallel(...)  # Generates data.csv and species.csv
    
    # Train neural network on the data
    make_model(input_specs, data_path, scaler_path, model_path)
```

---

### `make_data_parallel.py` - Training Data Generation

Generates autoignition simulation data using parallel processing and GPS-based species selection.

#### `make_data_parallel()`

```python
def make_data_parallel(
    fuel, mech_file, end_threshold, ign_HRR_threshold_div,
    ign_GPS_resolution, norm_GPS_resolution, GPS_per_interval,
    n_cases, t_rng, p_rng, phi_rng, alpha,
    always_threshold, never_threshold, pathname, species_ranges
)
```

**Purpose**: Run `n_cases` autoignition simulations with randomized initial conditions, apply GPS to identify important species at each interval, and save state vectors and species masks.

**Parameters**:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `fuel` | str | Fuel species name | `'CH4'` |
| `mech_file` | str | Cantera mechanism file | `'gri30.cti'` |
| `end_threshold` | float | HRR threshold to end simulation (J/m³s) | `2e5` |
| `ign_HRR_threshold_div` | int | Divisor for max HRR to define ignition | `300` |
| `ign_GPS_resolution` | int | Timesteps per interval during ignition | `200` |
| `norm_GPS_resolution` | int | Timesteps per interval post-ignition | `40` |
| `GPS_per_interval` | int | GPS evaluation points per interval | `4` |
| `n_cases` | int | Number of simulations to run | `100` |
| `t_rng` | list | Temperature range [min, max] (K) | `[800, 2300]` |
| `p_rng` | list | Log pressure range [min, max] (atm) | `[2.1, 2.5]` |
| `phi_rng` | list | Equivalence ratio range (not used if species_ranges set) | `[0.6, 1.4]` |
| `alpha` | float | GPS pathway threshold (smaller = more species) | `0.001` |
| `always_threshold` | float | Species occurrence threshold for "always include" | `0.99` |
| `never_threshold` | float | Species occurrence threshold for "never include" | `0.01` |
| `pathname` | str | Output directory for data | `'TrainingData/Data'` |
| `species_ranges` | dict | Species composition ranges | `{'CH4': (0, 1), 'O2': (0, 0.4)}` |

**Outputs**:
- `pathname/data.csv` - State vectors: [Temperature, Pressure, species mole fractions]
- `pathname/species.csv` - Binary masks: 1 if species important in that timestep
- `pathname/always_spec_nums.csv` - Indices of always-included species
- `pathname/never_spec_nums.csv` - Indices of never-included species

**Example**:
```python
make_data_parallel(
    fuel='CH4', mech_file='gri30.cti', end_threshold=2e5,
    ign_HRR_threshold_div=300, ign_GPS_resolution=200,
    norm_GPS_resolution=40, GPS_per_interval=4, n_cases=100,
    t_rng=[800, 2300], p_rng=[2.1, 2.5], phi_rng=[0.6, 1.4],
    alpha=0.001, always_threshold=0.99, never_threshold=0.01,
    pathname='TrainingData/CH4_data', species_ranges={
        'CH4': (0, 1), 'N2': (0, 0.8), 'O2': (0, 0.4)
    }
)
```

#### `process_simulation()` (internal)

```python
def process_simulation(sim_data) -> tuple[np.ndarray, np.ndarray]
```

Internal function called by joblib.Parallel for each simulation. Returns state vectors and species masks for one autoignition run.

---

### `mech_train.py` - Neural Network Training

Trains ensemble of ANNs to predict important species based on thermochemical state.

#### `spec_train()`

```python
def spec_train(X_train, Y_train) -> tuple[Model, History, History]
```

**Purpose**: Train a single ANN model for species prediction.

**Parameters**:
- `X_train` (np.ndarray): Normalized input data [samples, features]
- `Y_train` (np.ndarray): Binary target data [samples, species]

**Returns**:
- `model` - Trained Keras model
- `history` - Training history
- `train_test_history` - Train/test split history

**Default Architecture**:
```
Input Layer: [Temperature, Pressure, species mole fractions]
  ↓
Dense(16, activation='relu', kernel_initializer='he_normal')
  ↓
Dense(Y_train.shape[1], activation='sigmoid')
  ↓
Output Layer: [binary predictions for each species]
```

**To customize architecture, edit spec_train()** before the output layer:
```python
# Current default:
model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'))

# Example: Add deeper network
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'))
```

#### `make_model()`

```python
def make_model(input_specs, data_path, scaler_path, model_path)
```

**Purpose**: Load training data, normalize it, train ensemble of ANNs in parallel, and save best model.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_specs` | list | Species names used as ANN inputs | `['CH4', 'H2O', 'OH', 'O2', 'CO2']` |
| `data_path` | str | Directory containing data.csv and species.csv | `'TrainingData/Data'` |
| `scaler_path` | str | Path to save MinMaxScaler pickle | `'Scalers/scaler.pkl'` |
| `model_path` | str | Path to save best .h5 model | `'Models/model.h5'` |

**Key Settings**:
```python
num_processes = 28  # Parallel training (edit for your CPU cores)
train_test_split = 0.2  # 80% train, 20% validation
early_stopping_patience = 30  # Epochs to wait for val_loss improvement
batch_size = 32
epochs = 200
```

**Workflow**:
1. Load state vectors and species masks from CSV
2. Select only `input_specs` columns as features
3. Normalize input using MinMaxScaler (0-1 range)
4. Train 28 models in parallel with different random initializations
5. Select best model by validation loss
6. Save model as .h5 and scaler as .pkl

**Example**:
```python
make_model(
    input_specs=['CH4', 'H2O', 'OH', 'H', 'O2', 'CO2', 'O'],
    data_path='TrainingData/CH4_data',
    scaler_path='Scalers/ch4_scaler.pkl',
    model_path='Models/ch4_model.h5'
)
```

---

### `SL_GPS.py` - Adaptive Simulation

Runs autoignition simulation with trained ANN dynamically reducing the mechanism.

**Key Variables**:

```python
fuel = 'CH4'
mech_file = 'gri30.cti'  # Detailed mechanism
input_specs = ['CH4', 'H2O', 'OH', 'H', 'O2', 'CO2', 'O']  # ANN inputs
scaler_path = 'Scalers/scaler.pkl'  # MinMaxScaler
model_path = 'Models/model.h5'  # Trained ANN
data_path = 'TrainingData/Data'  # Training data (for always/never species)

T0_in = 1500  # Initial temperature (K)
phi = 1.0  # Equivalence ratio
atm = 1.0  # Pressure (atm)
t_end = 0.002  # Simulation end time (s)

norm_Dt = 0.0002  # Timestep pre-ignition (s)
ign_Dt = 0.00005  # Timestep during ignition (s)
ign_threshold = 9e7  # HRR threshold to detect ignition (J/m³s)

results_path = 'Results/results.pkl'  # Output pickle file
```

**Outputs**: Pickle file containing:
```python
{
    'time': array,  # Simulation time
    'temperature': array,  # Temperature evolution
    'heat_release_rate': array,  # HRR evolution
    'mole_fractions': dict,  # Species compositions
    'mechanism_size': array,  # Number of species/reactions over time
}
```

---

### `utils.py` - Core Utilities

Low-level functions for simulation, mechanism reduction, and GPS species selection.

#### `auto_ign_build_SL()`

```python
def auto_ign_build_SL(
    fuel, mech_file, input_specs, norm_Dt, ign_Dt,
    T0_in, phi, atm, t_end, scaler_path, model_path,
    data_path, ign_threshold
) -> dict
```

**Purpose**: Run adaptive autoignition simulation with ANN-based mechanism reduction.

**Workflow**:
1. Load trained ANN and scaler
2. Load training data to get always/never species
3. Start simulation at (T0_in, phi, atm)
4. At each timestep:
   - Get current thermochemical state
   - Scale state and feed to ANN
   - Get binary predictions for species
   - Combine with always/never species
   - Build reduced mechanism with only selected species
   - Step simulation
5. Switch between `norm_Dt` (slow) and `ign_Dt` (fast) based on HRR
6. Return results dict

#### `sub_mech()`

```python
def sub_mech(mech_file, species_names) -> ct.Solution
```

**Purpose**: Build a reduced Cantera solution with only specified species and their reactions.

**Parameters**:
- `mech_file` - Path to detailed CTI mechanism
- `species_names` - List of species to keep (must match Cantera names exactly)

**Returns**: Cantera Solution object with reduced mechanism

**Key Logic**:
- Keeps only reactions where ALL reactants and products are in species_names
- Removes reactions involving excluded species

#### `GPS_spec()`

```python
def GPS_spec(
    soln_in, fuel, raw, t_start, t_end, alpha, GPS_per_interval
) -> set
```

**Purpose**: Apply GPS algorithm to identify important species in a simulation interval.

**Parameters**:
- `soln_in` - Cantera solution object
- `fuel` - Fuel species name
- `raw` - Raw simulation data (time, T, P, X)
- `t_start`, `t_end` - Time interval (s)
- `alpha` - GPS pathway threshold (smaller = more species)
- `GPS_per_interval` - Number of GPS evaluations in interval

**Returns**: Set of important species

**GPS Configuration** (hardcoded in function):
```python
elements = ['C', 'H', 'O']
sources = [fuel, 'O2']
targets = ['CO2', 'H2O']
```

#### `findIgnInterval()`

```python
def findIgnInterval(hrr, threshold) -> tuple[int, int]
```

Returns time indices where HRR exceeds threshold (ignition start/end).

#### `find_ign_delay()`

```python
def find_ign_delay(times, temperature) -> float
```

Returns ignition delay time when temperature rises by 50% of (T_max - T_min).

#### `auto_ign_build_X0()`

```python
def auto_ign_build_X0(
    soln, T0, atm, X0, end_threshold=2e3, end=5, dir_raw=None
) -> tuple[dict, float, int]
```

**Purpose**: Run detailed autoignition with full mechanism.

**Parameters**:
- `soln` - Cantera solution
- `T0` - Initial temperature (K)
- `atm` - Pressure (atm)
- `X0` - Initial composition (mole fractions dict or string)
- `end_threshold` - HRR limit to stop simulation
- `end` - Maximum simulation time (s)
- `dir_raw` - Directory to save raw data

**Returns**:
- `raw` - Simulation data dict
- `time_exec` - CPU execution time (s)
- `step_count` - Number of timesteps

---

## Usage Examples

### Example 1: Generate Training Data Only

```python
from slgps.make_data_parallel import make_data_parallel

make_data_parallel(
    fuel='CH4',
    mech_file='gri30.cti',
    end_threshold=2e5,
    ign_HRR_threshold_div=300,
    ign_GPS_resolution=200,
    norm_GPS_resolution=40,
    GPS_per_interval=4,
    n_cases=50,  # Small for testing
    t_rng=[1000, 2000],
    p_rng=[2.0, 2.5],
    phi_rng=[0.8, 1.2],
    alpha=0.001,
    always_threshold=0.99,
    never_threshold=0.01,
    pathname='MyData',
    species_ranges={'CH4': (0, 1), 'O2': (0, 0.4), 'N2': (0, 0.8)}
)
```

### Example 2: Train ANN on Existing Data

```python
from slgps.mech_train import make_model

make_model(
    input_specs=['CH4', 'O2', 'CO2', 'H2O'],
    data_path='MyData',
    scaler_path='MyScaler.pkl',
    model_path='MyModel.h5'
)
```

### Example 3: Run Adaptive Simulation

```python
from slgps.utils import auto_ign_build_SL

results = auto_ign_build_SL(
    fuel='CH4',
    mech_file='gri30.cti',
    input_specs=['CH4', 'O2', 'CO2', 'H2O'],
    norm_Dt=0.0002,
    ign_Dt=0.00005,
    T0_in=1500,
    phi=1.0,
    atm=1.0,
    t_end=0.001,
    scaler_path='MyScaler.pkl',
    model_path='MyModel.h5',
    data_path='MyData',
    ign_threshold=9e7
)
```

---

## File Formats

### `data.csv` (State Vectors)

```
# Temperature,Atmospheres,CH4,O2,N2,CO2,H2O,...
1500,1.0,0.05,0.21,0.74,0.0,0.0,...
1550,1.0,0.04,0.20,0.76,0.001,0.01,...
...
```

Column order:
1. `# Temperature` (K)
2. `Atmospheres` (pressure, atm)
3. Species mole fractions in order of `input_specs`

### `species.csv` (Binary Masks)

```
CH4,O2,N2,CO2,H2O,...
1,1,1,0,0,...
1,1,1,1,1,...
...
```

Values: 1 = important in this timestep, 0 = not important

---

## Performance Tips

- **Reduce GPS_per_interval** from 4 to 2 for faster data generation
- **Reduce n_cases** for initial testing
- **Use fewer input_specs** to speed up ANN training
- **Increase num_processes** in mech_train.py if you have >28 CPU cores
- **Use GPU**: Install tensorflow[and-cuda] for 10-100x training speedup

---

## Debugging

Enable verbose output:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # See TensorFlow logs
import tensorflow as tf
tf.debugging.set_log_device_placement(True)  # Log device usage
```

Check intermediate results:

```python
import pandas as pd
data = pd.read_csv('MyData/data.csv')
print(data.head())
print(data.shape)  # (num_samples, num_features)
```
