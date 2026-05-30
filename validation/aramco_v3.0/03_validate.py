"""
Step 3: Validate SL-GPS Reduced Mechanism — AramcoMech 3.0
===========================================================
Runs the trained ANN on multiple validation cases spanning the
temperature-pressure space, and compares with detailed mechanism results.

Input:  models/ch4/model.h5, models/ch4/scaler.pkl
Output: results/ch4/*.pkl (one per validation case)
"""

import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from slgps.utils import auto_ign_build_SL, auto_ign_build_X0
import cantera as ct

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fuel & mechanism
fuel = 'CH4'
mech_file = os.path.join(os.path.dirname(__file__), 'mechanism', 'aramco_v3.cti')

# Trained model paths
model_path = os.path.join(os.path.dirname(__file__), 'models', 'ch4', 'model.h5')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'ch4', 'scaler.pkl')
data_path = os.path.join(os.path.dirname(__file__), 'data', 'train')

# Input species (must match training)
input_specs = [
    'CH4', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
    'CH3', 'HO2', 'CH2O', 'C2H6', 'C2H4', 'C2H2'
]

# SL-GPS simulation parameters
ign_threshold = 9e7
norm_Dt = 0.0002
ign_Dt = 0.00005

# Output directory
results_dir = os.path.join(os.path.dirname(__file__), 'results', 'ch4')

# Validation cases: (label, T0_K, P_atm, phi, t_end)
validation_cases = [
    ('low_T_low_P',   1000,  1.0, 1.0, 0.5),
    ('low_T_mid_P',   1000, 10.0, 1.0, 0.1),
    ('low_T_high_P',  1000, 40.0, 1.0, 0.05),
    ('mid_T_low_P',   1500,  1.0, 1.0, 0.01),
    ('mid_T_mid_P',   1500, 10.0, 1.0, 0.005),
    ('mid_T_high_P',  1500, 40.0, 1.0, 0.002),
    ('high_T_low_P',  2000,  1.0, 1.0, 0.005),
    ('high_T_high_P', 2000, 40.0, 1.0, 0.001),
    ('lean_mid',      1200, 10.0, 0.6, 0.1),
    ('rich_mid',      1200, 10.0, 1.5, 0.1),
]

# ============================================================================
# EXECUTION
# ============================================================================

def run_detailed_simulation(soln, T0, atm, phi, t_end):
    """Run a detailed (full mechanism) autoignition simulation."""
    soln_copy = ct.Solution(mech_file)
    soln_copy.TP = T0, atm * 101325
    soln_copy.set_equivalence_ratio(phi, f'{fuel}:1.0', 'O2:1.0, N2:3.76')
    X0 = dict(zip(soln_copy.species_names, soln_copy.X))

    result = auto_ign_build_X0(soln_copy, T0, atm, X0, end_threshold=None, end=t_end, dir_raw='ign')
    return result


def run_slgps_simulation(T0, atm, phi, t_end):
    """Run the SL-GPS reduced mechanism simulation."""
    result = auto_ign_build_SL(
        fuel, mech_file, input_specs, norm_Dt, ign_Dt,
        T0, phi, atm, t_end,
        scaler_path, model_path, data_path, ign_threshold
    )
    return result


if __name__ == '__main__':
    # Verify prerequisites
    if not os.path.isfile(mech_file):
        print(f"ERROR: Mechanism file not found: {mech_file}")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"ERROR: Trained model not found: {model_path}")
        print("Run 02_train_model.py first.")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("SL-GPS Validation — AramcoMech 3.0")
    print("=" * 70)
    print(f"  Validation cases: {len(validation_cases)}")
    print("=" * 70)

    soln = ct.Solution(mech_file)

    for i, (label, T0, P_atm, phi, t_end) in enumerate(validation_cases):
        print(f"\n{'─' * 50}")
        print(f"  Case {i+1}/{len(validation_cases)}: {label}")
        print(f"  T={T0} K, P={P_atm} atm, φ={phi}, t_end={t_end} s")
        print(f"{'─' * 50}")

        case_results = {}

        # Run detailed simulation
        print("  Running detailed simulation...")
        try:
            det_result = run_detailed_simulation(soln, T0, P_atm, phi, t_end)
            case_results['detailed'] = det_result
            print(f"    ✓ Detailed: {len(det_result[0]['axis0'])} timesteps")
        except Exception as e:
            print(f"    ✗ Detailed simulation failed: {e}")
            case_results['detailed'] = None

        # Run SL-GPS simulation
        print("  Running SL-GPS simulation...")
        try:
            sl_result = run_slgps_simulation(T0, P_atm, phi, t_end)
            case_results['slgps'] = sl_result
            print(f"    ✓ SL-GPS: complete")
        except Exception as e:
            print(f"    ✗ SL-GPS simulation failed: {e}")
            case_results['slgps'] = None

        # Save results
        case_results['config'] = {
            'label': label, 'T0': T0, 'P_atm': P_atm,
            'phi': phi, 't_end': t_end
        }

        result_file = os.path.join(results_dir, f'{label}.pkl')
        with open(result_file, 'wb') as f:
            pickle.dump(case_results, f)
        print(f"  Saved: {result_file}")

    print("\n" + "=" * 70)
    print("✓ All validation cases complete.")
    print(f"  Results saved to: {results_dir}")
    print("=" * 70)
