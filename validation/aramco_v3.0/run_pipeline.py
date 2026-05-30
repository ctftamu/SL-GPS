"""
SL-GPS Full Validation Pipeline — AramcoMech 3.0
==================================================
Self-contained script that runs:
  1. Training data generation (GPS)
  2. Neural network training
  3. Validation simulations (detailed vs SL-GPS)
  4. Plot generation
  5. LaTeX report compilation

AramcoMech 3.0: 581 species, 3036 reactions (C1-C4 combustion)
"""

import sys
import os
import time
import pickle
import numpy as np

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, '..', '..', 'src')
sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data', 'train')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'ch4')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'ch4')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'ch4')
DOCS_DIR = BASE_DIR

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MECH_FILE = os.path.join(BASE_DIR, 'mechanism', 'aramco_v3.cti')
FUEL = 'CH4'

# Training parameters
N_CASES = 10  # Reduced due to large mechanism (581 species)
T_RNG = [900, 2000]
P_RNG = [0.0, 1.3]         # 1-20 atm (log10)
PHI_RNG = [0.6, 1.4]
ALPHA = 0.001

SPECIES_RANGES = {
    'CH4': (0.02, 0.10),
    'O2':  (0.10, 0.25),
    'N2':  (0.60, 0.80),
    'CO2': (0.00, 0.005),
    'H2O': (0.00, 0.03),
}

ALWAYS_THRESHOLD = 0.99
NEVER_THRESHOLD = 0.01

# Input species (key intermediates for CH4 in Aramco 3.0)
INPUT_SPECS = [
    'CH4', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
    'CH3', 'HO2', 'CH2O', 'C2H6', 'C2H4', 'C2H2'
]

# Validation cases (fewer and shorter due to computational cost)
VALIDATION_CASES = [
    ('T1000_P1',   1000,  1.0, 1.0, 0.1),
    ('T1500_P1',   1500,  1.0, 1.0, 0.005),
    ('T1500_P10',  1500, 10.0, 1.0, 0.002),
    ('T2000_P1',   2000,  1.0, 1.0, 0.001),
]

# ============================================================================
# STEP 1: GENERATE TRAINING DATA
# ============================================================================

def step1_generate_data():
    print("\n" + "=" * 70)
    print("  STEP 1: Training Data Generation (GPS)")
    print("=" * 70)

    if os.path.isfile(os.path.join(DATA_DIR, 'data.csv')):
        print("  Data already exists, skipping generation.")
        return True

    from slgps.make_data_parallel import make_data_parallel

    print(f"  Mechanism: AramcoMech 3.0 (581 species, 3036 reactions)")
    print(f"  Simulations: {N_CASES}")
    print(f"  T range: {T_RNG[0]}-{T_RNG[1]} K")
    print(f"  P range: {10**P_RNG[0]:.1f}-{10**P_RNG[1]:.1f} atm")
    print(f"  NOTE: This will take significantly longer than GRI-Mech due to mechanism size.")

    t0 = time.time()
    make_data_parallel(
        fuel=FUEL,
        mech_file=MECH_FILE,
        end_threshold=2e5,
        ign_HRR_threshold_div=300,
        ign_GPS_resolution=100,   # Reduced for speed
        norm_GPS_resolution=20,   # Reduced for speed
        GPS_per_interval=2,       # Reduced for speed
        n_cases=N_CASES,
        t_rng=T_RNG,
        p_rng=P_RNG,
        phi_rng=PHI_RNG,
        alpha=ALPHA,
        always_threshold=ALWAYS_THRESHOLD,
        never_threshold=NEVER_THRESHOLD,
        pathname=DATA_DIR,
        species_ranges=SPECIES_RANGES
    )
    elapsed = time.time() - t0
    print(f"\n  ✓ Data generation complete ({elapsed:.1f}s / {elapsed/60:.1f} min)")

    for f in ['data.csv', 'species.csv']:
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp):
            lines = sum(1 for _ in open(fp)) - 1
            print(f"    {f}: {lines} samples")
    return True


# ============================================================================
# STEP 2: TRAIN NEURAL NETWORK
# ============================================================================

def step2_train():
    print("\n" + "=" * 70)
    print("  STEP 2: Neural Network Training")
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
    print(f"    Model: {model_path}")
    print(f"    Scaler: {scaler_path}")
    return True


# ============================================================================
# STEP 3: VALIDATION SIMULATIONS
# ============================================================================

def step3_validate():
    print("\n" + "=" * 70)
    print("  STEP 3: Validation Simulations")
    print("=" * 70)

    import cantera as ct
    from slgps.utils import auto_ign_build_SL, auto_ign_build_X0

    model_path = os.path.join(MODELS_DIR, 'model.h5')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    print(f"  Cases: {len(VALIDATION_CASES)}")

    for i, (label, T0, P_atm, phi, t_end) in enumerate(VALIDATION_CASES):
        result_file = os.path.join(RESULTS_DIR, f'{label}.pkl')
        if os.path.isfile(result_file):
            print(f"  [{i+1}/{len(VALIDATION_CASES)}] {label}: already exists, skipping")
            continue

        print(f"  [{i+1}/{len(VALIDATION_CASES)}] {label}: T={T0}K, P={P_atm}atm, φ={phi}")
        case_results = {'config': {'label': label, 'T0': T0, 'P_atm': P_atm, 'phi': phi, 't_end': t_end}}

        # Detailed simulation
        try:
            soln = ct.Solution(MECH_FILE)
            soln.TP = T0, P_atm * 101325
            soln.set_equivalence_ratio(phi, f'{FUEL}:1.0', 'O2:1.0, N2:3.76')
            X0 = dict(zip(soln.species_names, soln.X))
            det_result = auto_ign_build_X0(soln, T0, P_atm, X0, end_threshold=None, end=t_end, dir_raw='ign')
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

    print(f"\n  ✓ All validation cases complete")
    return True


# ============================================================================
# STEP 4: GENERATE PLOTS
# ============================================================================

def step4_plots():
    print("\n" + "=" * 70)
    print("  STEP 4: Plot Generation")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cantera as ct

    result_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.pkl')])
    if not result_files:
        print("  No results to plot!")
        return False

    all_cases = []
    for fname in result_files:
        with open(os.path.join(RESULTS_DIR, fname), 'rb') as f:
            all_cases.append(pickle.load(f))

    soln = ct.Solution(MECH_FILE)
    full_n_spec = len(soln.species_names)
    full_n_rxn = len(soln.reactions())

    # Individual case plots
    for case_data in all_cases:
        config = case_data['config']
        label = config['label']
        det = case_data.get('detailed')
        sl = case_data.get('slgps')

        if det is None and sl is None:
            continue

        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(
            f"AramcoMech 3.0 — {label}\n"
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
        plot_specs = ['CH4', 'O2', 'CO2', 'H2O', 'CO', 'OH']
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

    # Summary plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels_list = [c['config']['label'] for c in all_cases]
    x = np.arange(len(labels_list))

    # Ignition delay
    ign_det = []
    ign_sl = []
    for case_data in all_cases:
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
    ax1.set_title('Ignition Delay — AramcoMech 3.0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_list, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Reduction
    avg_specs = []
    for case_data in all_cases:
        sl = case_data.get('slgps')
        if sl is not None and len(sl) > 3:
            avg_specs.append(np.mean(sl[3]))
        else:
            avg_specs.append(0)

    ax2.bar(x, avg_specs, color='green', alpha=0.7)
    ax2.axhline(full_n_spec, color='k', ls='--', lw=1.5, label=f'Full ({full_n_spec} species)')
    ax2.set_ylabel('Avg Species in Reduced Mechanism')
    ax2.set_title('Mechanism Reduction — AramcoMech 3.0')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_list, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary.png")

    print(f"\n  ✓ All plots saved to {PLOTS_DIR}")
    return True


# ============================================================================
# STEP 5: UPDATE LATEX REPORT
# ============================================================================

def step5_report():
    print("\n" + "=" * 70)
    print("  STEP 5: LaTeX Report")
    print("=" * 70)

    # The report.tex already exists from earlier creation.
    # Try to compile it.
    tex_path = os.path.join(DOCS_DIR, 'report.tex')
    if not os.path.isfile(tex_path):
        print(f"  report.tex not found at {tex_path}")
        return True

    import subprocess
    try:
        subprocess.run(['pdflatex', '--version'], capture_output=True, check=True)
        for _ in range(2):
            subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'],
                         cwd=DOCS_DIR, capture_output=True)
        if os.path.isfile(os.path.join(DOCS_DIR, 'report.pdf')):
            print(f"  ✓ PDF compiled: {os.path.join(DOCS_DIR, 'report.pdf')}")
        else:
            print(f"  PDF compilation had issues. Compile manually.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  pdflatex not available. Compile manually: cd {DOCS_DIR} && pdflatex report.tex")

    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  SL-GPS VALIDATION — AramcoMech 3.0 (581 species, 3036 reactions)  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not os.path.isfile(MECH_FILE):
        print(f"\n  ERROR: Mechanism file not found: {MECH_FILE}")
        sys.exit(1)

    overall_start = time.time()

    steps = [
        ("Data Generation", step1_generate_data),
        ("NN Training", step2_train),
        ("Validation", step3_validate),
        ("Plotting", step4_plots),
        ("Report", step5_report),
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
    print(f"\n{'═' * 70}")
    print(f"  ✓ FULL PIPELINE COMPLETE — {total:.1f}s ({total/60:.1f} min)")
    print(f"{'═' * 70}")
