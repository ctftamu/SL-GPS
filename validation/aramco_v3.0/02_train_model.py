"""
Step 2: Train Neural Network on AramcoMech 3.0 GPS Data
========================================================
Trains an ensemble of ANNs to predict species importance from state vectors.

Input:  data/train/data.csv, data/train/species.csv
Output: models/ch4/model.h5, models/ch4/scaler.pkl
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from slgps.mech_train import make_model

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to training data (output of step 1)
data_path = os.path.join(os.path.dirname(__file__), 'data', 'train')

# Output paths for model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'models', 'ch4', 'model.h5')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'ch4', 'scaler.pkl')

# Input species for ANN (state vector components beyond T and P)
# These are key combustion intermediates/products for CH4 oxidation in Aramco 3.0
input_specs = [
    'CH4', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
    'CH3', 'HO2', 'CH2O', 'C2H6', 'C2H4', 'C2H2'
]

# ANN architecture
num_hidden_layers = 2
neurons_per_layer = 32

# Number of parallel training runs (ensemble)
num_processes = 16

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Verify training data exists
    data_csv = os.path.join(data_path, 'data.csv')
    species_csv = os.path.join(data_path, 'species.csv')

    if not os.path.isfile(data_csv) or not os.path.isfile(species_csv):
        print(f"ERROR: Training data not found at: {data_path}")
        print("Run 01_generate_data.py first.")
        sys.exit(1)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    print("=" * 70)
    print("SL-GPS Neural Network Training — AramcoMech 3.0")
    print("=" * 70)
    print(f"  Training data:   {data_path}")
    print(f"  Input species:   {input_specs}")
    print(f"  Architecture:    {num_hidden_layers} hidden layers × {neurons_per_layer} neurons")
    print(f"  Ensemble size:   {num_processes}")
    print(f"  Model output:    {model_path}")
    print(f"  Scaler output:   {scaler_path}")
    print("=" * 70)

    make_model(
        input_specs=input_specs,
        data_path=data_path,
        scaler_path=scaler_path,
        model_path=model_path,
        num_hidden_layers=num_hidden_layers,
        neurons_per_layer=neurons_per_layer,
        num_processes=num_processes
    )

    print("\n✓ Model training complete.")
    print(f"  Model saved to:  {model_path}")
    print(f"  Scaler saved to: {scaler_path}")
