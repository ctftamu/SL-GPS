"""
SL-GPS Validation — ThInK 1.0 with Methyl Formate (CH3OCHO)
=============================================================
Runs the full pipeline for methyl formate fuel:
  1. GPS training data generation
  2. Neural network training
  3. Validation (detailed vs SL-GPS)
  4. Plot generation + detailed report

ThInK 1.0: 160 species, 1089 reactions
Fuel: CH3OCHO (methyl formate, C2H4O2, MW=60.05)
"""

import sys
import os
import time
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, '..', '..', 'src')
sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data', 'train_ch3ocho')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'ch3ocho')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'ch3ocho')

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MECH_FILE = os.path.join(BASE_DIR, 'ThInK_1.0_Mech.yaml')
FUEL = 'CH3OCHO'

N_CASES = 15
T_RNG = [800, 1600]         # Wide range covering NTC and high-T
P_RNG = [0.0, 1.3]          # 1-20 atm (log10)
PHI_RNG = [0.7, 1.4]
ALPHA = 0.001

# Methyl formate / air composition ranges
SPECIES_RANGES = {
    'CH3OCHO': (0.01, 0.05),
    'O2':      (0.10, 0.25),
    'N2':      (0.60, 0.80),
    'CO2':     (0.00, 0.005),
    'H2O':     (0.00, 0.02),
}

ALWAYS_THRESHOLD = 0.99
NEVER_THRESHOLD = 0.01

# Input species for methyl formate combustion (fuel + key intermediates)
INPUT_SPECS = [
    'CH3OCHO', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
    'CH4', 'CH2O', 'CH3OH', 'HO2', 'CH3O', 'HCO'
]

# Validation cases (label, T0, P_atm, phi, t_end)
# Designed from ignition delay scoping:
#   800K/10atm ~45ms, 1000K/1atm ~36ms, 1200K/1atm ~2.2ms, 1600K/1atm ~0.08ms
VALIDATION_CASES = [
    # --- Baseline cases ---
    ('CH3OCHO_T1000_P1',    1000,  1.0, 1.0, 0.2),
    ('CH3OCHO_T1200_P1',    1200,  1.0, 1.0, 0.02),
    ('CH3OCHO_T1200_P10',   1200, 10.0, 1.0, 0.005),
    ('CH3OCHO_T1500_P1',    1500,  1.0, 1.0, 0.005),
    # --- More challenging cases ---
    ('CH3OCHO_T800_P20',     800, 20.0, 1.0, 0.5),      # Low-T, high-P
    ('CH3OCHO_T900_P10',     900, 10.0, 1.0, 0.1),       # Low-T, moderate-P
    ('CH3OCHO_T1100_P1_rich', 1100, 1.0, 1.3, 0.05),    # Rich mixture
    ('CH3OCHO_T1400_P5_lean', 1400, 5.0, 0.7, 0.005),   # Lean mixture
]


# ============================================================================
# STEP 1: GENERATE TRAINING DATA
# ============================================================================

def step1_generate_data():
    print("\n" + "=" * 70)
    print("  STEP 1: Training Data Generation (CH3OCHO / GPS)")
    print("=" * 70)

    if os.path.isfile(os.path.join(DATA_DIR, 'data.csv')):
        print("  Data already exists, skipping generation.")
        return True

    from slgps.make_data_parallel import make_data_parallel

    print(f"  Fuel: CH3OCHO (methyl formate)")
    print(f"  Mechanism: ThInK 1.0 ({160} species, {1089} reactions)")
    print(f"  Simulations: {N_CASES} (PARALLEL — all available cores)")
    print(f"  T range: {T_RNG[0]}-{T_RNG[1]} K")
    print(f"  P range: {10**P_RNG[0]:.1f}-{10**P_RNG[1]:.1f} atm")

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
    print("  STEP 2: Neural Network Training (CH3OCHO)")
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
    print("  STEP 3: Validation Simulations (CH3OCHO)")
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
        case_results = {'config': {'label': label, 'T0': T0, 'P_atm': P_atm, 'phi': phi, 't_end': t_end, 'fuel': 'CH3OCHO'}}

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

    print(f"\n  ✓ All CH3OCHO validation cases complete")
    return True


# ============================================================================
# STEP 4: GENERATE PLOTS
# ============================================================================

def step4_plots():
    print("\n" + "=" * 70)
    print("  STEP 4: Plot Generation (CH3OCHO)")
    print("=" * 70)

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
        print("  No results to plot!")
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
            f"ThInK 1.0 — Methyl Formate (CH₃OCHO) — {label}\n"
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
        plot_specs = ['CH3OCHO', 'O2', 'CO2', 'H2O', 'CO', 'CH2O', 'CH3OH']
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
        plt.savefig(os.path.join(RESULTS_DIR, f'{label}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {label}.png")

    # Summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ThInK 1.0 — Methyl Formate (CH₃OCHO) Validation Summary', fontsize=12, fontweight='bold')

    labels_list = [c['config']['label'].replace('CH3OCHO_', '') for c in cases]
    x = np.arange(len(labels_list))

    # Ignition delay comparison
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
    ax1.set_xticklabels(labels_list, rotation=45, ha='right')
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
    ax2.set_xticklabels(labels_list, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'CH3OCHO_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: CH3OCHO_summary.png")

    print(f"\n  ✓ All CH3OCHO plots saved")
    return True


# ============================================================================
# STEP 5: GENERATE REPORT
# ============================================================================

def step5_report():
    print("\n" + "=" * 70)
    print("  STEP 5: Detailed Report (CH3OCHO)")
    print("=" * 70)

    import cantera as ct

    soln = ct.Solution(MECH_FILE)
    full_n_spec = len(soln.species_names)
    full_n_rxn = len(soln.reaction_equations())

    # Load results
    all_cases = []
    for label, *_ in VALIDATION_CASES:
        fpath = os.path.join(RESULTS_DIR, f'{label}.pkl')
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                all_cases.append(pickle.load(f))

    # Compute metrics
    metrics = []
    for case_data in all_cases:
        config = case_data['config']
        det = case_data.get('detailed')
        sl = case_data.get('slgps')

        def get_ign_delay(result):
            if result is None:
                return None
            t = np.array(result[0]['axis0'])
            T = np.array(result[0]['temperature'])
            if len(t) < 3:
                return None
            dTdt = np.diff(T) / np.diff(t)
            return t[np.argmax(dTdt)]

        ign_det = get_ign_delay(det)
        ign_sl = get_ign_delay(sl)

        ign_err = None
        if ign_det and ign_sl and ign_det > 0:
            ign_err = abs(ign_sl - ign_det) / ign_det * 100

        avg_spec = None
        avg_rxn = None
        if sl is not None and len(sl) > 3:
            avg_spec = np.mean(sl[3])
            avg_rxn = np.mean(sl[2])

        passed = ign_err is not None and ign_err < 15

        metrics.append({
            'label': config['label'],
            'T0': config['T0'],
            'P_atm': config['P_atm'],
            'phi': config['phi'],
            'ign_det_ms': ign_det * 1000 if ign_det else None,
            'ign_sl_ms': ign_sl * 1000 if ign_sl else None,
            'ign_err_pct': ign_err,
            'avg_spec': avg_spec,
            'avg_rxn': avg_rxn,
            'spec_reduction_pct': (1 - avg_spec / full_n_spec) * 100 if avg_spec else None,
            'passed': passed,
        })

    # Load training data stats
    n_train_samples = 0
    n_var_species = 0
    n_always = 0
    n_never = 0
    data_csv = os.path.join(DATA_DIR, 'data.csv')
    if os.path.isfile(data_csv):
        n_train_samples = sum(1 for _ in open(data_csv)) - 1
    for name, attr in [('var_spec_nums.csv', 'n_var'), ('always_spec_nums.csv', 'n_always'), ('never_spec_nums.csv', 'n_never')]:
        fp = os.path.join(DATA_DIR, name)
        if os.path.isfile(fp):
            import pandas as pd
            df = pd.read_csv(fp)
            count = len(df.columns) - 1 if len(df.columns) > 1 else 0
            if 'var' in name:
                n_var_species = count
            elif 'always' in name:
                n_always = count
            elif 'never' in name:
                n_never = count

    # Print report
    report_lines = []
    def rprint(s=''):
        report_lines.append(s)
        print(s)

    rprint()
    rprint("=" * 78)
    rprint("  SL-GPS VALIDATION REPORT")
    rprint("  Mechanism: ThInK 1.0 | Fuel: Methyl Formate (CH3OCHO)")
    rprint("=" * 78)
    rprint()
    rprint("  MECHANISM DETAILS")
    rprint(f"    Name:        ThInK 1.0")
    rprint(f"    Species:     {full_n_spec}")
    rprint(f"    Reactions:   {full_n_rxn}")
    rprint(f"    Fuel:        CH3OCHO (methyl formate, C2H4O2, MW=60.05)")
    rprint()
    rprint("  TRAINING CONFIGURATION")
    rprint(f"    Simulations: {N_CASES}")
    rprint(f"    T range:     {T_RNG[0]}-{T_RNG[1]} K")
    rprint(f"    P range:     {10**P_RNG[0]:.0f}-{10**P_RNG[1]:.0f} atm")
    rprint(f"    φ range:     {PHI_RNG[0]}-{PHI_RNG[1]}")
    rprint(f"    α (GPS):     {ALPHA}")
    rprint(f"    Samples:     {n_train_samples}")
    rprint(f"    Variable:    {n_var_species} species (ANN-predicted)")
    rprint(f"    Always:      {n_always} species (always included)")
    rprint(f"    Never:       {n_never} species (always excluded)")
    rprint(f"    Input specs: {INPUT_SPECS}")
    rprint(f"    NN arch:     2 × 32 neurons, ReLU, sigmoid output")
    rprint()
    rprint("  VALIDATION RESULTS")
    rprint(f"  {'Case':<28s} {'T(K)':>5s} {'P(atm)':>7s} {'φ':>4s}  {'τ_det(ms)':>10s} {'τ_SL(ms)':>10s} {'Error%':>7s} {'Avg Sp':>7s} {'Red%':>5s} {'Pass':>5s}")
    rprint("  " + "-" * 96)

    n_passed = 0
    ign_errors = []
    reductions = []
    avg_specs_list = []

    for m in metrics:
        det_str = f"{m['ign_det_ms']:.3f}" if m['ign_det_ms'] else "N/A"
        sl_str = f"{m['ign_sl_ms']:.3f}" if m['ign_sl_ms'] else "N/A"
        err_str = f"{m['ign_err_pct']:.1f}" if m['ign_err_pct'] is not None else "N/A"
        sp_str = f"{m['avg_spec']:.0f}" if m['avg_spec'] else "N/A"
        red_str = f"{m['spec_reduction_pct']:.0f}" if m['spec_reduction_pct'] else "N/A"
        pass_str = "✓" if m['passed'] else "✗"

        short_label = m['label'].replace('CH3OCHO_', '')
        rprint(f"  {short_label:<28s} {m['T0']:>5d} {m['P_atm']:>7.1f} {m['phi']:>4.1f}  {det_str:>10s} {sl_str:>10s} {err_str:>7s} {sp_str:>7s} {red_str:>5s} {pass_str:>5s}")

        if m['passed']:
            n_passed += 1
        if m['ign_err_pct'] is not None:
            ign_errors.append(m['ign_err_pct'])
        if m['spec_reduction_pct'] is not None:
            reductions.append(m['spec_reduction_pct'])
        if m['avg_spec'] is not None:
            avg_specs_list.append(m['avg_spec'])

    rprint("  " + "-" * 96)
    rprint()
    rprint("  SUMMARY")
    rprint(f"    Cases passed:        {n_passed}/{len(metrics)}")
    if ign_errors:
        rprint(f"    Mean ign. error:     {np.mean(ign_errors):.1f}%")
        rprint(f"    Max ign. error:      {np.max(ign_errors):.1f}%")
    if reductions:
        rprint(f"    Mean reduction:      {np.mean(reductions):.0f}%")
        rprint(f"    Mean species count:  {np.mean(avg_specs_list):.0f} / {full_n_spec}")
    rprint()
    rprint("=" * 78)

    # Save report to file
    report_path = os.path.join(RESULTS_DIR, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Report saved: {report_path}")
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  SL-GPS VALIDATION — ThInK 1.0 / Methyl Formate (CH3OCHO)         ║")
    print("║  160 species, 1089 reactions                                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not os.path.isfile(MECH_FILE):
        print(f"\n  ERROR: Mechanism file not found: {MECH_FILE}")
        sys.exit(1)

    overall_start = time.time()

    steps = [
        ("Data Generation (CH3OCHO)", step1_generate_data),
        ("NN Training (CH3OCHO)", step2_train),
        ("Validation (CH3OCHO)", step3_validate),
        ("Plotting (CH3OCHO)", step4_plots),
        ("Report (CH3OCHO)", step5_report),
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
    print(f"  ✓ CH3OCHO PIPELINE COMPLETE — {total:.1f}s ({total/60:.1f} min)")
    print(f"{'═' * 70}")
