"""
SL-GPS Validation — AramcoMech 3.0 with Ethylene (C2H4)
========================================================
Runs the full pipeline for ethylene fuel:
  1. GPS training data generation
  2. Neural network training
  3. Validation (detailed vs SL-GPS)
  4. Plot generation

AramcoMech 3.0: 581 species, 3036 reactions
Fuel: C2H4 (ethylene)
"""

import sys
import os
import time
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, '..', '..', 'src')
sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data', 'train_c2h4')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'c2h4')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'c2h4')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'c2h4')

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MECH_FILE = os.path.join(BASE_DIR, 'mechanism', 'aramco_v3.cti')
FUEL = 'C2H4'

N_CASES = 15
T_RNG = [900, 1900]
P_RNG = [0.0, 1.3]         # 1-20 atm (log10)
PHI_RNG = [0.6, 1.5]
ALPHA = 0.001

# Ethylene/air composition ranges
SPECIES_RANGES = {
    'C2H4': (0.02, 0.10),
    'O2':   (0.10, 0.25),
    'N2':   (0.60, 0.80),
    'CO2':  (0.00, 0.005),
    'H2O':  (0.00, 0.02),
}

ALWAYS_THRESHOLD = 0.99
NEVER_THRESHOLD = 0.01

# Input species for ethylene combustion
INPUT_SPECS = [
    'C2H4', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
    'CH4', 'C2H2', 'C2H6', 'CH2O', 'HO2', 'CH3'
]

# Validation cases for ethylene (label, T0, P_atm, phi, t_end)
VALIDATION_CASES = [
    # --- Baseline cases ---
    ('C2H4_T1000_P1',      1000,  1.0, 1.0, 0.1),
    ('C2H4_T1200_P10',     1200, 10.0, 1.0, 0.01),
    ('C2H4_T1500_P1',      1500,  1.0, 1.0, 0.003),
    ('C2H4_T1500_P10',     1500, 10.0, 1.0, 0.001),
    # --- More difficult cases ---
    ('C2H4_T900_P20',       900, 20.0, 1.0, 0.5),     # Low-T, high-P
    ('C2H4_T1100_P1_rich', 1100,  1.0, 1.4, 0.1),    # Rich mixture
    ('C2H4_T1300_P5_lean', 1300,  5.0, 0.7, 0.005),  # Lean mixture
    ('C2H4_T1800_P20',    1800, 20.0, 1.0, 0.0005),  # High-T, high-P
]


# ============================================================================
# STEP 1: GENERATE TRAINING DATA
# ============================================================================

def step1_generate_data():
    print("\n" + "=" * 70)
    print("  STEP 1: Training Data Generation (C2H4 / GPS)")
    print("=" * 70)

    if os.path.isfile(os.path.join(DATA_DIR, 'data.csv')):
        print("  Data already exists, skipping generation.")
        return True

    from slgps.make_data_parallel import process_simulation
    import random
    import cantera as ct
    import pandas as pd

    print(f"  Fuel: C2H4 (ethylene)")
    print(f"  Mechanism: AramcoMech 3.0 (581 species, 3036 reactions)")
    print(f"  Simulations: {N_CASES} (SERIAL — avoids fork deadlock)")
    print(f"  T range: {T_RNG[0]}-{T_RNG[1]} K")
    print(f"  P range: {10**P_RNG[0]:.1f}-{10**P_RNG[1]:.1f} atm")
    print(f"  α (GPS threshold): {ALPHA}")

    t0 = time.time()

    soln_in = ct.Solution(MECH_FILE)
    det_spec_strs = list(soln_in.species_names)

    # Generate random initial conditions
    temps = [random.uniform(T_RNG[0], T_RNG[1]) for _ in range(N_CASES)]
    pressures = [10**(random.uniform(P_RNG[0], P_RNG[1])) for _ in range(N_CASES)]

    X0_values = []
    for _ in range(N_CASES):
        species_values = {sp: random.uniform(r[0], r[1]) for sp, r in SPECIES_RANGES.items()}
        total = sum(species_values.values())
        if total > 0:
            species_values = {sp: v/total for sp, v in species_values.items()}
        X0_values.append(', '.join(f'{sp}:{v:.5f}' for sp, v in species_values.items()))

    # Run simulations serially
    all_data = []
    all_bins = []
    for i in range(N_CASES):
        sim_data = (temps[i], X0_values[i], pressures[i], 5.0,
                    100, 20, 300, det_spec_strs, MECH_FILE, FUEL,
                    2e5, ALPHA, 2, i)
        print(f"    Simulation {i+1}/{N_CASES}: T={temps[i]:.0f}K, P={pressures[i]:.1f}atm")
        result = process_simulation(sim_data)
        if result is not None:
            all_data.append(result[0])
            all_bins.append(result[1])
            print(f"      ✓ {result[0].shape[0]} samples")
        else:
            print(f"      ✗ Failed")

    if not all_data:
        print("  ERROR: No simulations succeeded!")
        return False

    # Combine and save
    data_combined = np.vstack(all_data)
    bin_combined = np.vstack([b.reshape(-1, b.shape[-1]) for b in all_bins])

    # Save data.csv
    header = '# Temperature,Atmospheres,' + ','.join(det_spec_strs)
    np.savetxt(os.path.join(DATA_DIR, 'data.csv'), data_combined, delimiter=',', header=header, comments='')

    # Save species.csv
    header_sp = ','.join(det_spec_strs) + ',end'
    np.savetxt(os.path.join(DATA_DIR, 'species.csv'), bin_combined, delimiter=',', header=header_sp, comments='# ', fmt='%d')

    # Compute always/never/variable species
    n_samples = bin_combined.shape[0]
    freq = bin_combined.sum(axis=0) / n_samples

    always_specs = [det_spec_strs[j] for j in range(len(det_spec_strs)) if freq[j] >= ALWAYS_THRESHOLD]
    never_specs = [det_spec_strs[j] for j in range(len(det_spec_strs)) if freq[j] <= NEVER_THRESHOLD]
    var_specs = [det_spec_strs[j] for j in range(len(det_spec_strs)) if NEVER_THRESHOLD < freq[j] < ALWAYS_THRESHOLD]

    pd.DataFrame(columns=always_specs + ['end']).to_csv(os.path.join(DATA_DIR, 'always_spec_nums.csv'), index=False)
    pd.DataFrame(columns=never_specs + ['end']).to_csv(os.path.join(DATA_DIR, 'never_spec_nums.csv'), index=False)
    pd.DataFrame(columns=var_specs + ['end']).to_csv(os.path.join(DATA_DIR, 'var_spec_nums.csv'), index=False)

    # Update species.csv to only include variable species
    var_indices = [det_spec_strs.index(sp) for sp in var_specs]
    bin_var = bin_combined[:, var_indices]
    header_var = ','.join(var_specs) + ',end'
    np.savetxt(os.path.join(DATA_DIR, 'species.csv'), bin_var, delimiter=',', header=header_var, comments='# ', fmt='%d')

    elapsed = time.time() - t0
    print(f"\n  ✓ Data generation complete ({elapsed:.1f}s / {elapsed/60:.1f} min)")
    print(f"    Total samples: {n_samples}")
    print(f"    Always species: {len(always_specs)}")
    print(f"    Variable species: {len(var_specs)}")
    print(f"    Never species: {len(never_specs)}")
    return True


# ============================================================================
# STEP 2: TRAIN NEURAL NETWORK
# ============================================================================

def step2_train():
    print("\n" + "=" * 70)
    print("  STEP 2: Neural Network Training (C2H4)")
    print("=" * 70)

    model_path = os.path.join(MODELS_DIR, 'model.h5')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    if os.path.isfile(model_path):
        print("  Model already exists, skipping training.")
        return True

    from slgps.mech_train import make_model

    print(f"  Input species: {INPUT_SPECS}")
    print(f"  Architecture: 2 hidden layers × 32 neurons")
    print(f"  Ensemble: 8 parallel trainings")

    t0 = time.time()
    make_model(
        input_specs=INPUT_SPECS,
        data_path=DATA_DIR,
        scaler_path=scaler_path,
        model_path=model_path,
        num_hidden_layers=2,
        neurons_per_layer=32,
        num_processes=8
    )
    elapsed = time.time() - t0
    print(f"\n  ✓ Training complete ({elapsed:.1f}s)")
    return True


# ============================================================================
# STEP 3: VALIDATION SIMULATIONS
# ============================================================================

def step3_validate():
    print("\n" + "=" * 70)
    print("  STEP 3: Validation Simulations (C2H4)")
    print("=" * 70)

    import cantera as ct
    from slgps.utils import auto_ign_build_SL, auto_ign_build_X0

    model_path = os.path.join(MODELS_DIR, 'model.h5')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    for i, (label, T0, P_atm, phi, t_end) in enumerate(VALIDATION_CASES):
        result_file = os.path.join(RESULTS_DIR, f'{label}.pkl')
        if os.path.isfile(result_file):
            print(f"  [{i+1}/{len(VALIDATION_CASES)}] {label}: already exists, skipping")
            continue

        print(f"  [{i+1}/{len(VALIDATION_CASES)}] {label}: T={T0}K, P={P_atm}atm, φ={phi}")
        case_results = {'config': {'label': label, 'T0': T0, 'P_atm': P_atm, 'phi': phi, 't_end': t_end, 'fuel': 'C2H4'}}

        # Detailed simulation
        try:
            soln = ct.Solution(MECH_FILE)
            soln.TP = T0, P_atm * 101325
            soln.set_equivalence_ratio(phi, f'{FUEL}:1.0', 'O2:1.0, N2:3.76')
            X0_str = ', '.join(f'{sp}:{x:.6f}' for sp, x in zip(soln.species_names, soln.X) if x > 1e-10)
            det_result = auto_ign_build_X0(soln, T0, P_atm, X0_str, end_threshold=None, end=t_end, dir_raw='ign')
            case_results['detailed'] = det_result
            print(f"      Detailed: ✓ ({len(det_result[0]['axis0'])} steps)")
        except Exception as e:
            print(f"      Detailed: ✗ ({e})")
            case_results['detailed'] = None

        # SL-GPS simulation
        try:
            sl_result = auto_ign_build_SL(
                FUEL, MECH_FILE, INPUT_SPECS, 0.0002, 0.00005,
                T0, phi, P_atm, t_end,
                scaler_path, model_path, DATA_DIR, 9e7
            )
            case_results['slgps'] = sl_result
            print(f"      SL-GPS:   ✓")
        except Exception as e:
            print(f"      SL-GPS:   ✗ ({e})")
            case_results['slgps'] = None

        with open(result_file, 'wb') as f:
            pickle.dump(case_results, f)

    print(f"\n  ✓ All C2H4 validation cases complete")
    return True


# ============================================================================
# STEP 4: GENERATE PLOTS
# ============================================================================

def step4_plots():
    print("\n" + "=" * 70)
    print("  STEP 4: Plot Generation (C2H4)")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cantera as ct

    soln = ct.Solution(MECH_FILE)
    full_n_spec = len(soln.species_names)

    # Load results
    cases = []
    for label, *_ in VALIDATION_CASES:
        fpath = os.path.join(RESULTS_DIR, f'{label}.pkl')
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                cases.append(pickle.load(f))

    if not cases:
        print("  No C2H4 results to plot!")
        return False

    # Individual case plots
    for case_data in cases:
        config = case_data['config']
        label = config['label']
        det = case_data.get('detailed')
        sl = case_data.get('slgps')

        if det is None and sl is None:
            continue

        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(
            f"AramcoMech 3.0 — Ethylene (C₂H₄) — {label}\n"
            f"T₀={config['T0']}K, P={config['P_atm']}atm, φ={config['phi']}",
            fontsize=12, fontweight='bold'
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
            axs[1].plot(np.array(det[0]['axis0'])*1000, det[0]['heat_release_rate'], 'k-', lw=1.5, label='Detailed')
        if sl is not None:
            axs[1].plot(np.array(sl[0]['axis0'])*1000, sl[0]['heat_release_rate'], 'r--', lw=1.5, label='SL-GPS')
        axs[1].set_ylabel('HRR (W/m³)')
        axs[1].set_yscale('symlog', linthresh=1e4)
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # Species
        plot_specs = ['C2H4', 'O2', 'CO2', 'H2O', 'CO', 'C2H2', 'CH4']
        colors = plt.cm.tab10(np.linspace(0, 1, len(plot_specs)))
        for j, spec in enumerate(plot_specs):
            if det is not None:
                try:
                    idx = soln.species_names.index(spec)
                    mf = np.array(det[0]['mole_fraction'])
                    if len(mf.shape) == 2 and idx < mf.shape[1]:
                        axs[2].plot(np.array(det[0]['axis0'])*1000, mf[:, idx], '-', color=colors[j], lw=1.2, label=f'{spec}')
                except (ValueError, IndexError):
                    pass
        axs[2].set_ylabel('Mole Fraction')
        axs[2].legend(ncol=3, fontsize=8)
        axs[2].grid(True, alpha=0.3)

        # Mechanism size
        if sl is not None and len(sl) > 3:
            mech_times = np.array(sl[1]) * 1000
            axs[3].plot(mech_times, sl[3], 'g-o', ms=2, lw=1, label='# Species')
            ax2 = axs[3].twinx()
            ax2.plot(mech_times, sl[2], 'b-s', ms=2, lw=1, label='# Reactions')
            ax2.set_ylabel('# Reactions', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            axs[3].axhline(full_n_spec, color='g', ls=':', alpha=0.5, label=f'Full ({full_n_spec} sp)')
        axs[3].set_ylabel('# Species', color='g')
        axs[3].tick_params(axis='y', labelcolor='g')
        axs[3].set_xlabel('Time (ms)')
        axs[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{label}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {label}.png")

    # Summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('AramcoMech 3.0 — Ethylene (C₂H₄) Validation Summary', fontsize=12, fontweight='bold')

    labels_list = [c['config']['label'].replace('C2H4_', '') for c in cases]
    x = np.arange(len(labels_list))

    # Ignition delay
    ign_det = []
    ign_sl = []
    for case_data in cases:
        def get_ign_delay(result):
            if result is None:
                return 0
            t = np.array(result[0]['axis0'])
            T = np.array(result[0]['temperature'])
            if len(t) < 3:
                return 0
            dTdt = np.diff(T) / np.diff(t)
            return t[np.argmax(dTdt)] * 1000
        ign_det.append(get_ign_delay(case_data.get('detailed')))
        ign_sl.append(get_ign_delay(case_data.get('slgps')))

    width = 0.35
    ax1.bar(x - width/2, ign_det, width, label='Detailed', color='black', alpha=0.7)
    ax1.bar(x + width/2, ign_sl, width, label='SL-GPS', color='red', alpha=0.7)
    ax1.set_ylabel('Ignition Delay (ms)')
    ax1.set_title('Ignition Delay')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Reduction
    avg_specs = []
    for case_data in cases:
        sl = case_data.get('slgps')
        if sl is not None and len(sl) > 3:
            avg_specs.append(np.mean(sl[3]))
        else:
            avg_specs.append(0)

    ax2.bar(x, avg_specs, color='green', alpha=0.7)
    ax2.axhline(full_n_spec, color='k', ls='--', lw=1.5, label=f'Full ({full_n_spec} species)')
    ax2.set_ylabel('Avg Species in Reduced Mechanism')
    ax2.set_title('Mechanism Reduction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'C2H4_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: C2H4_summary.png")

    print(f"\n  ✓ All C2H4 plots saved")
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  SL-GPS VALIDATION — AramcoMech 3.0 / Ethylene (C2H4)             ║")
    print("║  581 species, 3036 reactions                                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not os.path.isfile(MECH_FILE):
        print(f"\n  ERROR: Mechanism file not found: {MECH_FILE}")
        sys.exit(1)

    overall_start = time.time()

    steps = [
        ("Data Generation (C2H4)", step1_generate_data),
        ("NN Training (C2H4)", step2_train),
        ("Validation (C2H4)", step3_validate),
        ("Plotting (C2H4)", step4_plots),
    ]

    for name, func in steps:
        try:
            success = func()
            if not success:
                print(f"\n  ✗ {name} failed!")
                sys.exit(1)
        except Exception as e:
            print(f"\n  ✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    total = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*70}")
