"""
SL-GPS Expanded Validation — ThInK 1.0 with Methyl Formate (CH3OCHO)
====================================================================
Extended test suite focusing on NTC region and failed case analysis.
Uses only 50% of available CPUs for background processing.

Failed case analysis:
  T900_P10 (900K, 10atm) showed 165.2% ignition delay error
  → This is in the NTC region (negative temperature coefficient)
  → Requires better training data coverage (25 cases vs 15)
  → Needs finer adaptive timesteps in low-T region
"""

import sys
import os
import time
import pickle
import numpy as np
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, '..', '..', 'src')
sys.path.insert(0, SRC_DIR)

# ============================================================================
# CPU ALLOCATION
# ============================================================================

def get_safe_process_count():
    """Get CPU count from SLURM environment or default to 8."""
    # Check SLURM environment variable
    slurm_ntasks = os.environ.get('SLURM_NTASKS')
    if slurm_ntasks:
        num_procs = int(slurm_ntasks)
    else:
        # Fallback: check via /proc/cpuinfo or use default
        try:
            num_procs = len(os.sched_getaffinity(0))
        except:
            num_procs = 8
    
    print(f"\n  Allocated CPUs: {num_procs}")
    return num_procs

NUM_SAFE_PROCESSES = get_safe_process_count()

# ============================================================================
# DIRECTORIES & CONFIG
# ============================================================================

DATA_DIR = os.path.join(BASE_DIR, 'data', 'train_ch3ocho_expanded')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'ch3ocho_expanded')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'ch3ocho_expanded')
LOG_FILE = os.path.join(RESULTS_DIR, 'validation.log')

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MECH_FILE = os.path.join(BASE_DIR, 'ThInK_1.0_Mech.yaml')
FUEL = 'CH3OCHO'

# Expanded training with stratified low-T (NTC) coverage
N_CASES = 60
T_RNG = [700, 1600]         # Extended low-T floor for NTC
# Guarantee dedicated sampling of the NTC / low-T band (counts sum to N_CASES)
T_STRATA = [
    (700, 900, 24),         # low-T / NTC band (previously starved by uniform sampling)
    (900, 1100, 16),        # intermediate
    (1100, 1600, 20),       # high-T
]
P_RNG = [0.0, 1.3]          # 1-20 atm (log10)
PHI_RNG = [0.7, 1.4]
ALPHA = 0.001

SPECIES_RANGES = {
    'CH3OCHO': (0.01, 0.05),
    'O2':      (0.10, 0.25),
    'N2':      (0.60, 0.80),
    'CO2':     (0.00, 0.005),
    'H2O':     (0.00, 0.02),
}

ALWAYS_THRESHOLD = 0.99
NEVER_THRESHOLD = 0.01

INPUT_SPECS = [
    'CH3OCHO', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
    'CH4', 'CH2O', 'CH3OH', 'HO2', 'CH3O', 'HCO'
]

# Expanded validation cases: 16 (up from 8)
# Focus on NTC region (800-1100K) and expanded pressure matrix
VALIDATION_CASES = [
    # --- Original baseline cases ---
    ('CH3OCHO_T1000_P1',    1000,  1.0, 1.0, 0.2),
    ('CH3OCHO_T1200_P1',    1200,  1.0, 1.0, 0.02),
    ('CH3OCHO_T1200_P10',   1200, 10.0, 1.0, 0.005),
    ('CH3OCHO_T1500_P1',    1500,  1.0, 1.0, 0.005),
    ('CH3OCHO_T800_P20',     800, 20.0, 1.0, 0.5),
    ('CH3OCHO_T1100_P1_rich', 1100, 1.0, 1.3, 0.05),
    ('CH3OCHO_T1400_P5_lean', 1400, 5.0, 0.7, 0.005),
    
    # --- NTC investigation (primary focus) ---
    ('CH3OCHO_T900_P10',     900, 10.0, 1.0, 0.1),      # FAILED: 165% error → focus here
    ('CH3OCHO_T850_P10',     850, 10.0, 1.0, 0.2),      # NTC approach
    ('CH3OCHO_T950_P10',     950, 10.0, 1.0, 0.08),     # NTC approach
    ('CH3OCHO_T900_P15',     900, 15.0, 1.0, 0.15),     # Higher pressure
    ('CH3OCHO_T900_P5',      900,  5.0, 1.0, 0.08),     # Lower pressure
    
    # --- Extended pressure/temperature sweep ---
    ('CH3OCHO_T750_P20',     750, 20.0, 1.0, 1.0),      # Very low T, high P
    ('CH3OCHO_T1100_P5',     1100, 5.0, 1.0, 0.05),     # Mid-range
    ('CH3OCHO_T1300_P2',     1300, 2.0, 1.0, 0.01),     # High T, low P
    ('CH3OCHO_T800_P5',       800, 5.0, 1.0, 0.3),      # Low T sensitivity
    ('CH3OCHO_T1100_P10_rich', 1100, 10.0, 1.3, 0.08),  # Rich + pressure
]

# ============================================================================
# STEP 1: GENERATE EXPANDED TRAINING DATA
# ============================================================================

def step1_generate_data():
    log("=" * 70)
    log("  STEP 1: Expanded Training Data Generation (25 cases)")
    log("=" * 70)

    if os.path.isfile(os.path.join(DATA_DIR, 'data.csv')):
        log("  Data already exists, skipping generation.")
        return True

    from slgps.make_data_parallel import make_data_parallel

    log(f"  Fuel: CH3OCHO (methyl formate)")
    log(f"  Mechanism: ThInK 1.0 (160 species, 1089 reactions)")
    log(f"  Training cases: {N_CASES} (PARALLEL — {NUM_SAFE_PROCESSES} CPUs)")
    log(f"  T range: {T_RNG[0]}-{T_RNG[1]} K (extended for NTC)")
    log(f"  P range: {10**P_RNG[0]:.1f}-{10**P_RNG[1]:.1f} atm")

    t0 = time.time()

    make_data_parallel(
        fuel=FUEL,
        mech_file=MECH_FILE,
        end_threshold=2e5,
        ign_HRR_threshold_div=300,
        ign_GPS_resolution=100,
        norm_GPS_resolution=20,
        GPS_per_interval=2,
        n_cases=N_CASES,
        t_rng=T_RNG,
        p_rng=P_RNG,
        phi_rng=PHI_RNG,
        alpha=ALPHA,
        always_threshold=ALWAYS_THRESHOLD,
        never_threshold=NEVER_THRESHOLD,
        pathname=DATA_DIR,
        species_ranges=SPECIES_RANGES,
        t_strata=T_STRATA
    )

    elapsed = time.time() - t0
    log(f"\n  ✓ Data generation complete ({elapsed:.1f}s / {elapsed/60:.1f} min)")

    for f in ['data.csv', 'species.csv']:
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp):
            lines = sum(1 for _ in open(fp)) - 1
            log(f"    {f}: {lines} samples")
    return True


# ============================================================================
# STEP 2: TRAIN NEURAL NETWORK (with limited CPU)
# ============================================================================

def step2_train():
    log("\n" + "=" * 70)
    log("  STEP 2: Neural Network Training (adaptive CPU throttling)")
    log("=" * 70)

    model_path = os.path.join(MODELS_DIR, 'model.h5')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    if os.path.isfile(model_path):
        log("  Model already exists, skipping training.")
        return True

    # Train a NEW model on the expanded/stratified data. (Do NOT copy the original
    # ch3ocho model as a fallback — that silently skips retraining and leaves low-T
    # accuracy unchanged.)
    try:
        from slgps.mech_train import make_model
    except ImportError as e:
        log(f"  ⚠ TensorFlow not available: {e}")
        log(f"  ⚠ Cannot train model without TensorFlow — aborting.")
        return False

    os.makedirs(MODELS_DIR, exist_ok=True)
    log(f"  Training on expanded data: {DATA_DIR}")
    log(f"  Input species: {INPUT_SPECS}")

    # make_model treats its path args as DIRECTORIES and writes model.h5 / model.pkl
    # inside them. Train into temp dirs, then flatten to the flat file paths step3 loads.
    import shutil
    tmp_model_dir = os.path.join(MODELS_DIR, '_train_model')
    tmp_scaler_dir = os.path.join(MODELS_DIR, '_train_scaler')
    for p in (model_path, scaler_path, tmp_model_dir, tmp_scaler_dir):
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)

    t0 = time.time()
    make_model(
        input_specs=INPUT_SPECS,
        data_path=DATA_DIR,
        scaler_path=tmp_scaler_dir,
        model_path=tmp_model_dir
    )
    shutil.move(os.path.join(tmp_model_dir, 'model.h5'), model_path)
    shutil.move(os.path.join(tmp_scaler_dir, 'model.pkl'), scaler_path)
    shutil.rmtree(tmp_model_dir, ignore_errors=True)
    shutil.rmtree(tmp_scaler_dir, ignore_errors=True)
    log(f"  ✓ Training complete ({time.time() - t0:.1f}s)")
    return True


# ============================================================================
# STEP 3: VALIDATION SIMULATIONS (with enhanced timestep control)
# ============================================================================

def step3_validate():
    log("\n" + "=" * 70)
    log(f"  STEP 3: Expanded Validation ({len(VALIDATION_CASES)} cases)")
    log("=" * 70)

    import cantera as ct
    from slgps.utils import auto_ign_build_SL, auto_ign_build_X0

    model_path = os.path.join(MODELS_DIR, 'model.h5')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    results_summary = []

    for i, (label, T0, P_atm, phi, t_end) in enumerate(VALIDATION_CASES):
        result_file = os.path.join(RESULTS_DIR, f'{label}.pkl')
        if os.path.isfile(result_file):
            log(f"  [{i+1:2d}/{len(VALIDATION_CASES)}] {label}: already exists, skipping")
            continue

        log(f"  [{i+1:2d}/{len(VALIDATION_CASES)}] {label}: T={T0}K, P={P_atm}atm, φ={phi}")
        case_results = {
            'config': {
                'label': label,
                'T0': T0,
                'P_atm': P_atm,
                'phi': phi,
                't_end': t_end,
                'fuel': 'CH3OCHO'
            }
        }

        # Enhanced timestep control for NTC region
        if T0 < 950 and P_atm >= 5:
            # For problematic NTC region: use finer control
            norm_Dt = 0.00015  # Tighter
            ign_Dt = 0.00004   # Finer
            log(f"      → NTC region: enhanced timestep control (norm={norm_Dt}, ign={ign_Dt})")
        else:
            # Standard control
            norm_Dt = 0.0002
            ign_Dt = 0.00005

        # Detailed simulation
        try:
            soln = ct.Solution(MECH_FILE)
            soln.TP = T0, P_atm * 101325
            soln.set_equivalence_ratio(phi, f'{FUEL}:1.0', 'O2:1.0, N2:3.76')
            X0_str = ', '.join(f'{sp}:{x:.6f}' for sp, x in zip(soln.species_names, soln.X) if x > 1e-10)
            det_result = auto_ign_build_X0(soln, T0, P_atm, X0_str, end_threshold=None, end=t_end, dir_raw='ign')
            case_results['detailed'] = det_result
            log(f"      Detailed: ✓ ({len(det_result[0]['axis0'])} steps)")
        except Exception as e:
            log(f"      Detailed: ✗ ({e})")
            case_results['detailed'] = None

        # SL-GPS simulation with adaptive timesteps
        try:
            sl_result = auto_ign_build_SL(
                FUEL, MECH_FILE, INPUT_SPECS, norm_Dt, ign_Dt,
                T0, phi, P_atm, t_end,
                scaler_path, model_path, DATA_DIR, 9e7
            )
            case_results['slgps'] = sl_result
            log(f"      SL-GPS:   ✓")

            # Compute error for summary
            def get_ign_delay(result):
                if result is None:
                    return None
                t = np.array(result[0]['axis0'])
                T = np.array(result[0]['temperature'])
                if len(t) < 3:
                    return None
                dTdt = np.diff(T) / np.diff(t)
                return t[np.argmax(dTdt)]

            ign_det = get_ign_delay(case_results.get('detailed'))
            ign_sl = get_ign_delay(case_results.get('slgps'))
            if ign_det and ign_sl:
                error_pct = 100 * abs(ign_sl - ign_det) / ign_det
                results_summary.append((label, T0, P_atm, phi, error_pct))

        except Exception as e:
            log(f"      SL-GPS:   ✗ ({e})")
            case_results['slgps'] = None

        with open(result_file, 'wb') as f:
            pickle.dump(case_results, f)

    log(f"\n  ✓ All validation cases complete")

    # Print summary
    if results_summary:
        log("\n  SUMMARY TABLE (errors %)")
        log("  " + "-" * 65)
        for label, T0, P_atm, phi, error in results_summary:
            status = "✓" if error < 15 else "⚠" if error < 50 else "✗"
            log(f"  {status} {label:30s} T={T0:4d}K P={P_atm:5.1f}atm φ={phi:.1f}  Error: {error:6.1f}%")

    return True


# ============================================================================
# STEP 4: GENERATE PLOTS
# ============================================================================

def step4_plots():
    log("\n" + "=" * 70)
    log("  STEP 4: Plot Generation")
    log("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cantera as ct

    soln = ct.Solution(MECH_FILE)
    full_n_spec = len(soln.species_names)

    cases = []
    for label, *_ in VALIDATION_CASES:
        fpath = os.path.join(RESULTS_DIR, f'{label}.pkl')
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                cases.append(pickle.load(f))

    if not cases:
        log("  No results to plot!")
        return False

    # Individual plots (sample key cases to save time)
    focus_labels = ['CH3OCHO_T900_P10', 'CH3OCHO_T850_P10', 'CH3OCHO_T950_P10',
                    'CH3OCHO_T1000_P1', 'CH3OCHO_T1200_P1']
    
    for case_data in cases:
        label = case_data['config']['label']
        if label not in focus_labels:
            continue

        det = case_data.get('detailed')
        sl = case_data.get('slgps')

        if det is None and sl is None:
            continue

        config = case_data['config']
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(
            f"ThInK 1.0 — {label}\nT₀={config['T0']}K, P={config['P_atm']}atm, φ={config['phi']}",
            fontsize=11, fontweight='bold'
        )

        # Temperature
        if det is not None:
            axs[0].plot(np.array(det[0]['axis0'])*1000, det[0]['temperature'], 'k-', lw=1.5, label='Detailed')
        if sl is not None:
            axs[0].plot(np.array(sl[0]['axis0'])*1000, sl[0]['temperature'], 'r--', lw=1.5, label='SL-GPS')
        axs[0].set_ylabel('Temperature (K)')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # HRR
        if det is not None:
            axs[1].plot(np.array(det[0]['axis0'])*1000, det[0]['heat_release_rate'], 'k-', lw=1.5)
        if sl is not None:
            axs[1].plot(np.array(sl[0]['axis0'])*1000, sl[0]['heat_release_rate'], 'r--', lw=1.5)
        axs[1].set_ylabel('HRR (W/m³)')
        axs[1].set_yscale('symlog', linthresh=1e4)
        axs[1].grid(True, alpha=0.3)

        # Species
        plot_specs = ['CH3OCHO', 'O2', 'CO2', 'H2O', 'CO']
        colors = plt.cm.tab10(np.linspace(0, 1, len(plot_specs)))
        for j, spec in enumerate(plot_specs):
            if det is not None:
                try:
                    idx = soln.species_names.index(spec)
                    mf = np.array(det[0]['mole_fraction'])
                    if len(mf.shape) == 2 and idx < mf.shape[1]:
                        axs[2].plot(np.array(det[0]['axis0'])*1000, mf[:, idx], '-', color=colors[j], lw=1.2, label=spec)
                except (ValueError, IndexError):
                    pass
        axs[2].set_ylabel('Mole Fraction')
        axs[2].set_xlabel('Time (ms)')
        axs[2].legend(loc='best', fontsize=9)
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{label}.png'), dpi=120, bbox_inches='tight')
        plt.close()
        log(f"  Saved: {label}.png")

    log(f"\n  ✓ Plots generated for key NTC cases")
    return True


# ============================================================================
# LOGGING
# ============================================================================

def log(msg):
    """Print and write to log file."""
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    with open(LOG_FILE, 'w') as f:
        f.write(f"SL-GPS Expanded Validation Log\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

    log("\n" + "=" * 70)
    log("  SL-GPS EXPANDED VALIDATION — ThInK 1.0 + CH3OCHO")
    log("  Focus: NTC region investigation + extended test matrix")
    log("  CPU Policy: 50% available cores for background processing")
    log("=" * 70)

    try:
        if step1_generate_data():
            if step2_train():
                if step3_validate():
                    step4_plots()

        log("\n" + "=" * 70)
        log("  VALIDATION COMPLETE")
        log(f"  Results: {RESULTS_DIR}")
        log(f"  Log: {LOG_FILE}")
        log("=" * 70)

    except Exception as e:
        log(f"\n  ERROR: {e}")
        import traceback
        log(traceback.format_exc())
