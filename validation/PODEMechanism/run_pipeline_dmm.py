"""
SL-GPS Validation — PODE-1 (DMM) Mechanism with Dimethoxymethane (CH3OCH2OCH3)
================================================================================
Runs the full pipeline for DMM fuel:
  1. GPS training data generation
  2. Neural network training
  3. Validation (detailed vs SL-GPS)
  4. Plot generation + detailed report

PODE-1 Mechanism: 586 species, 3091 reactions (Labbé Lab)
Fuel: CH3OCH2OCH3 (dimethoxymethane / DMM / PODE-1, C3H8O2, MW=76.09)
"""

import sys
import os
import time
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, '..', '..', 'src')
sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data', 'train_dmm')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'dmm')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'dmm')

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MECH_FILE = os.path.join(BASE_DIR, 'mechanism', 'PODE-1_clean.yaml')
FUEL = 'CH3OCH2OCH3'

N_CASES = 15
T_RNG = [800, 1600]
P_RNG = [0.0, 1.3]          # 1-20 atm (log10)
PHI_RNG = [0.7, 1.4]
ALPHA = 0.001

SPECIES_RANGES = {
    'CH3OCH2OCH3': (0.01, 0.05),
    'O2':          (0.10, 0.25),
    'N2':          (0.60, 0.80),
    'CO2':         (0.00, 0.005),
    'H2O':         (0.00, 0.02),
}

ALWAYS_THRESHOLD = 0.99
NEVER_THRESHOLD = 0.01

INPUT_SPECS = [
    'CH3OCH2OCH3', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
    'CH4', 'CH2O', 'CH3OH', 'HO2', 'CH3O', 'HCO'
]

VALIDATION_CASES = [
    ('DMM_T1000_P1',    1000,  1.0, 1.0, 0.2),
    ('DMM_T1200_P1',    1200,  1.0, 1.0, 0.02),
    ('DMM_T1200_P10',   1200, 10.0, 1.0, 0.005),
    ('DMM_T1500_P1',    1500,  1.0, 1.0, 0.005),
    ('DMM_T800_P20',     800, 20.0, 1.0, 0.5),
    ('DMM_T900_P10',     900, 10.0, 1.0, 0.1),
    ('DMM_T1100_P1_rich', 1100, 1.0, 1.3, 0.05),
    ('DMM_T1400_P5_lean', 1400, 5.0, 0.7, 0.005),
]


# ============================================================================
# STEP 1: GENERATE TRAINING DATA
# ============================================================================

def step1_generate_data():
    print("\n" + "=" * 70)
    print("  STEP 1: Training Data Generation (DMM / GPS)")
    print("=" * 70)

    if os.path.isfile(os.path.join(DATA_DIR, 'data.csv')):
        print("  Data already exists, skipping generation.")
        return True

    from slgps.make_data_parallel import make_data_parallel

    print(f"  Fuel: CH3OCH2OCH3 (dimethoxymethane / DMM)")
    print(f"  Mechanism: PODE-1 (586 species, 3091 reactions)")
    print(f"  Simulations: {N_CASES} (PARALLEL)")
    print(f"  T range: {T_RNG[0]}-{T_RNG[1]} K")
    print(f"  P range: {10**P_RNG[0]:.1f}-{10**P_RNG[1]:.1f} atm")

    t0 = time.time()
    make_data_parallel(
        fuel=FUEL, mech_file=MECH_FILE,
        end_threshold=2e5, ign_HRR_threshold_div=300,
        ign_GPS_resolution=100, norm_GPS_resolution=20, GPS_per_interval=2,
        n_cases=N_CASES, t_rng=T_RNG, p_rng=P_RNG, phi_rng=PHI_RNG,
        alpha=ALPHA, always_threshold=ALWAYS_THRESHOLD,
        never_threshold=NEVER_THRESHOLD,
        pathname=DATA_DIR, species_ranges=SPECIES_RANGES,
    )
    elapsed = time.time() - t0
    print(f"\n  ✓ Data generation complete ({elapsed:.1f}s)")

    for f in ['data.csv', 'species.csv', 'always_spec_nums.csv',
              'never_spec_nums.csv', 'var_spec_nums.csv']:
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} — MISSING")
            return False
    return True


# ============================================================================
# STEP 2: TRAIN NEURAL NETWORK
# ============================================================================

def step2_train():
    print("\n" + "=" * 70)
    print("  STEP 2: Neural Network Training (DMM)")
    print("=" * 70)

    model_path = os.path.join(MODELS_DIR, 'model.h5')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    if os.path.isfile(model_path):
        print(f"  Model already exists: {model_path}")
        return True

    import tensorflow as tf
    from pandas import read_csv
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from joblib import Parallel, delayed
    from pickle import dump

    N_ENSEMBLE = 8

    def train_single(X, Y, seed):
        tf.keras.utils.set_random_seed(seed)
        Xtr, Xval, Ytr, Yval = train_test_split(
            X, Y, test_size=0.2, random_state=seed)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(Y.shape[1], activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=30,
            restore_best_weights=True)
        model.fit(Xtr, Ytr, epochs=200, batch_size=32, callbacks=[es],
                  verbose=0, validation_data=(Xval, Yval))
        return model, model.evaluate(Xval, Yval, verbose=0)

    X = read_csv(os.path.join(DATA_DIR, 'data.csv'))
    Y = read_csv(os.path.join(DATA_DIR, 'species.csv')).iloc[:, :-1]
    X = X[['# Temperature', 'Atmospheres'] + INPUT_SPECS]
    print(f"  Input: {X.shape[1]} features, Output: {Y.shape[1]} species")
    print(f"  Samples: {X.shape[0]}")

    scaler = preprocessing.MinMaxScaler()
    Xp = scaler.fit_transform(X)

    print(f"  Training {N_ENSEMBLE} ensemble models...")
    t0 = time.time()
    results = Parallel(n_jobs=min(N_ENSEMBLE, os.cpu_count() or 1))(
        delayed(train_single)(Xp, Y.to_numpy(), s)
        for s in range(N_ENSEMBLE))

    vals = [v for _, v in results]
    best = int(np.argmin(vals))
    print(f"  Val losses: {['%.4f' % v for v in vals]}")
    print(f"  Best: seed {best} (val_loss={vals[best]:.4f})")

    results[best][0].save(model_path, save_format='h5')
    with open(scaler_path, 'wb') as f:
        dump(scaler, f)

    print(f"  ✓ Training complete ({time.time()-t0:.1f}s)")
    return True


# ============================================================================
# STEP 3: VALIDATION
# ============================================================================

def step3_validate():
    print("\n" + "=" * 70)
    print("  STEP 3: Validation — Detailed vs SL-GPS (DMM)")
    print("=" * 70)

    import cantera as ct
    from slgps.utils import auto_ign_build_SL, auto_ign_build_X0

    MODEL = os.path.join(MODELS_DIR, 'model.h5')
    SCALER = os.path.join(MODELS_DIR, 'scaler.pkl')
    soln_ref = ct.Solution(MECH_FILE)
    full_n_spec = soln_ref.n_species

    summary = []
    for label, T0, P_atm, phi, t_end in VALIDATION_CASES:
        out = os.path.join(RESULTS_DIR, f'{label}.pkl')
        print(f'\n=== {label}  T={T0}K P={P_atm}atm phi={phi} ===', flush=True)
        case = {'config': dict(label=label, T0=T0, P_atm=P_atm, phi=phi, t_end=t_end)}

        t0 = time.time()
        soln = ct.Solution(MECH_FILE)
        soln.TP = T0, P_atm * 101325
        soln.set_equivalence_ratio(phi, f'{FUEL}:1.0', 'O2:1.0, N2:3.76')
        X0 = soln.mole_fraction_dict()
        case['detailed'] = auto_ign_build_X0(
            soln, T0, P_atm, X0, end_threshold=None, end=t_end, dir_raw='ign')
        print(f'  detailed done ({time.time()-t0:.0f}s)', flush=True)

        det_t = np.array(case['detailed'][0]['axis0'])
        det_T = np.array(case['detailed'][0]['temperature'])
        if len(det_t) < 3:
            print(f'  SKIPPED (too few steps)')
            continue
        idt_det = det_t[np.argmax(np.diff(det_T) / np.diff(det_t))]
        norm_Dt = min(2e-4, idt_det / 50)
        ign_Dt = min(idt_det / 200, norm_Dt / 4)
        print(f'  idt={idt_det:.3e}: norm_Dt={norm_Dt:.2e} ign_Dt={ign_Dt:.2e}', flush=True)

        t0 = time.time()
        case['slgps'] = auto_ign_build_SL(
            FUEL, MECH_FILE, INPUT_SPECS, norm_Dt, ign_Dt,
            T0, phi, P_atm, t_end, SCALER, MODEL, DATA_DIR, 9e7)
        print(f'  slgps done ({time.time()-t0:.0f}s)', flush=True)

        with open(out, 'wb') as f:
            pickle.dump(case, f)

        def ign_delay(r):
            if r is None: return None
            t = np.array(r[0]['axis0']); T = np.array(r[0]['temperature'])
            if len(t) < 3: return None
            return t[np.argmax(np.diff(T) / np.diff(t))]

        d, s = ign_delay(case['detailed']), ign_delay(case['slgps'])
        err = 100 * abs(s - d) / d if (d and s) else float('nan')
        nspec = np.mean(case['slgps'][3]) if case['slgps'] else float('nan')
        summary.append((label, T0, P_atm, phi, d, s, err, nspec))
        print(f'  idt: det={d:.4e}  sl={s:.4e}  err={err:.1f}%  avg_sp={nspec:.0f}', flush=True)

    print('\n' + '=' * 80)
    print(f'{"case":26s} {"T":>5} {"P":>5} {"phi":>4} {"det idt":>11} {"sl idt":>11} {"err%":>7} {"nsp":>4}')
    for label, T0, P, phi, d, s, err, nsp in summary:
        mark = 'PASS' if err < 10 else ('WARN' if err < 25 else 'FAIL')
        print(f'{label:26s} {T0:>5.0f} {P:>5.1f} {phi:>4.1f} {d:>11.4e} {s:>11.4e} {err:>6.1f}% {nsp:>4.0f} {mark}')
    return True



# ============================================================================
# STEP 4: PLOTS
# ============================================================================

def step4_plots():
    print("\n" + "=" * 70)
    print("  STEP 4: Plot Generation (DMM)")
    print("=" * 70)

    import cantera as ct
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    soln_ref = ct.Solution(MECH_FILE)
    full_n_spec = soln_ref.n_species
    species_names = soln_ref.species_names

    all_cases = []; labels_list = []
    ign_det_list = []; ign_sl_list = []; err_list = []

    def ign_delay(result):
        if result is None: return None
        t = np.array(result[0]['axis0']); T = np.array(result[0]['temperature'])
        if len(t) < 3: return None
        return t[np.argmax(np.diff(T) / np.diff(t))]

    for label, T0, P_atm, phi, t_end in VALIDATION_CASES:
        pkl = os.path.join(RESULTS_DIR, f'{label}.pkl')
        if not os.path.isfile(pkl):
            print(f"  SKIP: {label}"); continue
        with open(pkl, 'rb') as f:
            case = pickle.load(f)
        all_cases.append(case)
        labels_list.append(label.replace('DMM_', ''))

        det, sl = case.get('detailed'), case.get('slgps')
        d, s = ign_delay(det), ign_delay(sl)
        err = 100 * abs(s - d) / d if (d and s) else 0
        ign_det_list.append(d * 1000 if d else 0)
        ign_sl_list.append(s * 1000 if s else 0)
        err_list.append(err)

        cfg = case['config']
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f"PODE-1 — {label}\nT₀={cfg['T0']}K, P={cfg['P_atm']}atm, "
                     f"φ={cfg['phi']}  |  idt err {err:.1f}%",
                     fontsize=11, fontweight='bold')

        if det is not None:
            td = np.array(det[0]['axis0']) * 1000
            axs[0].plot(td, det[0]['temperature'], 'k-', lw=1.5, label='Detailed')
        if sl is not None:
            ts = np.array(sl[0]['axis0']) * 1000
            axs[0].plot(ts, sl[0]['temperature'], 'r--', lw=1.5, label='SL-GPS')
        axs[0].set_ylabel('Temperature (K)'); axs[0].legend(); axs[0].grid(True, alpha=0.3)

        if det is not None:
            axs[1].plot(td, det[0]['heat_release_rate'], 'k-', lw=1.5)
        if sl is not None:
            axs[1].plot(ts, sl[0]['heat_release_rate'], 'r--', lw=1.5)
        axs[1].set_ylabel('HRR (W/m³)')
        axs[1].set_yscale('symlog', linthresh=1e4); axs[1].grid(True, alpha=0.3)

        plot_specs = ['CH3OCH2OCH3', 'O2', 'CO2', 'H2O', 'CO']
        colors = plt.cm.tab10(np.linspace(0, 1, len(plot_specs)))
        for j, spec in enumerate(plot_specs):
            if det is not None:
                try:
                    idx = species_names.index(spec)
                    mf = np.array(det[0]['mole_fraction'])
                    if mf.ndim == 2 and idx < mf.shape[1]:
                        axs[2].plot(td, mf[:, idx], '-', color=colors[j], lw=1.2, label=spec)
                except (ValueError, IndexError):
                    pass
        axs[2].set_ylabel('Mole Fraction'); axs[2].set_xlabel('Time (ms)')
        axs[2].legend(loc='best', fontsize=9); axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{label}.png'), dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {label}.png")

    # Summary plot
    if all_cases:
        x = np.arange(len(labels_list))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
        fig.suptitle('PODE-1 / DMM — SL-GPS Validation Summary',
                     fontsize=12, fontweight='bold')
        w = 0.38
        ax1.bar(x - w/2, ign_det_list, w, label='Detailed', color='0.25')
        ax1.bar(x + w/2, ign_sl_list, w, label='SL-GPS', color='tab:red', alpha=0.85)
        for i, e in enumerate(err_list):
            ax1.text(i, max(ign_det_list[i], ign_sl_list[i])*1.25,
                     f'{e:.1f}%', ha='center', fontsize=7)
        ax1.set_yscale('log'); ax1.set_ylabel('Ignition delay (ms)')
        ax1.set_xticks(x); ax1.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)
        ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)
        ax1.set_title('Ignition delay: detailed vs SL-GPS')

        avg_specs = []
        for c in all_cases:
            sl = c.get('slgps')
            avg_specs.append(np.mean(sl[3]) if (sl and len(sl) > 3) else 0)
        ax2.bar(x, avg_specs, color='tab:blue', alpha=0.85)
        ax2.axhline(full_n_spec, color='k', ls='--', lw=1.5,
                    label=f'Full ({full_n_spec} species)')
        ax2.set_ylabel('Avg Species in Reduced Mechanism')
        ax2.set_title('Mechanism Reduction — PODE-1')
        ax2.set_xticks(x); ax2.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)
        ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'DMM_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: DMM_summary.png")

    print(f"\n  ✓ All plots saved to {RESULTS_DIR}")
    return True



# ============================================================================
# STEP 5: REPORT
# ============================================================================

def step5_report():
    print("\n" + "=" * 70)
    print("  STEP 5: Validation Report (DMM)")
    print("=" * 70)

    import cantera as ct
    soln_ref = ct.Solution(MECH_FILE)
    full_n_spec = soln_ref.n_species
    report_lines = []

    def rprint(msg=''):
        report_lines.append(msg); print(msg)

    def ign_delay(result):
        if result is None: return None
        t = np.array(result[0]['axis0']); T = np.array(result[0]['temperature'])
        if len(t) < 3: return None
        return t[np.argmax(np.diff(T) / np.diff(t))]

    rprint()
    rprint("=" * 78)
    rprint("  SL-GPS VALIDATION REPORT — PODE-1 / DMM (CH3OCH2OCH3)")
    rprint(f"  Mechanism: PODE-1 ({full_n_spec} species, 3091 reactions)")
    rprint("=" * 78)
    rprint()

    metrics = []
    for label, T0, P_atm, phi, t_end in VALIDATION_CASES:
        pkl = os.path.join(RESULTS_DIR, f'{label}.pkl')
        if not os.path.isfile(pkl): continue
        with open(pkl, 'rb') as f:
            case = pickle.load(f)
        det, sl = case.get('detailed'), case.get('slgps')
        d, s = ign_delay(det), ign_delay(sl)
        err = 100 * abs(s - d) / d if (d and s) else None
        avg_spec = np.mean(sl[3]) if (sl and len(sl) > 3) else None
        red = 100 * (1 - avg_spec / full_n_spec) if avg_spec else None
        metrics.append({
            'label': label, 'T0': T0, 'P_atm': P_atm, 'phi': phi,
            'ign_det_ms': d * 1000 if d else None,
            'ign_sl_ms': s * 1000 if s else None,
            'ign_err_pct': err, 'avg_spec': avg_spec,
            'spec_reduction_pct': red,
            'passed': err is not None and err < 10,
        })

    rprint(f"  {'Case':<28s} {'T(K)':>5} {'P(atm)':>7} {'phi':>4}  "
           f"{'τ_det(ms)':>10s} {'τ_SL(ms)':>10s} {'Error%':>7s} "
           f"{'Avg Sp':>7s} {'Red%':>5s} {'Pass':>5s}")
    rprint("  " + "-" * 96)

    n_passed = 0; ign_errors = []; reductions = []; avg_specs_list = []
    for m in metrics:
        d_s = f"{m['ign_det_ms']:.3f}" if m['ign_det_ms'] else "N/A"
        s_s = f"{m['ign_sl_ms']:.3f}" if m['ign_sl_ms'] else "N/A"
        e_s = f"{m['ign_err_pct']:.1f}" if m['ign_err_pct'] is not None else "N/A"
        sp_s = f"{m['avg_spec']:.0f}" if m['avg_spec'] else "N/A"
        r_s = f"{m['spec_reduction_pct']:.0f}" if m['spec_reduction_pct'] else "N/A"
        p_s = "✓" if m['passed'] else "✗"
        short = m['label'].replace('DMM_', '')
        rprint(f"  {short:<28s} {m['T0']:>5d} {m['P_atm']:>7.1f} {m['phi']:>4.1f}  "
               f"{d_s:>10s} {s_s:>10s} {e_s:>7s} {sp_s:>7s} {r_s:>5s} {p_s:>5s}")
        if m['passed']: n_passed += 1
        if m['ign_err_pct'] is not None: ign_errors.append(m['ign_err_pct'])
        if m['spec_reduction_pct'] is not None: reductions.append(m['spec_reduction_pct'])
        if m['avg_spec'] is not None: avg_specs_list.append(m['avg_spec'])

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
    print("║  SL-GPS VALIDATION — PODE-1 / DMM (CH3OCH2OCH3)                   ║")
    print("║  586 species, 3091 reactions (Labbé Lab)                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not os.path.isfile(MECH_FILE):
        print(f"\n  ERROR: Mechanism file not found: {MECH_FILE}")
        sys.exit(1)

    overall_start = time.time()
    steps = [
        ("Data Generation (DMM)", step1_generate_data),
        ("NN Training (DMM)", step2_train),
        ("Validation (DMM)", step3_validate),
        ("Plotting (DMM)", step4_plots),
        ("Report (DMM)", step5_report),
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
    print(f"  ✓ DMM PIPELINE COMPLETE — {total:.1f}s ({total/60:.1f} min)")
    print(f"{'═' * 70}")

