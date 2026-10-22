"""
End-to-end from-scratch reproduction of the PODE-1 / DMM SL-GPS study.

Stages (each skipped if output exists, so the run is resumable):
  1. Generate four training batches (make_data_parallel)
  2. Build temperature-balanced training set (make_balanced_data.py)
  3. Never-list audit (audit_never_list.py) -- aborts on failure
  4. Train ANN ensemble, selecting by VALIDATION loss
  5. Full 35-case validation (validate_full.py)

Usage:
    python recreate_from_scratch.py            # full pipeline
    python recreate_from_scratch.py --tag NAME # custom model tag
"""
import argparse, os, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(BASE, '..', '..', 'src'))
sys.path.insert(0, SRC)
os.chdir(BASE)

MECH = os.path.join('mechanism', 'PODE-1_clean.yaml')
FUEL = 'CH3OCH2OCH3'
INPUT_SPECS = ['CH3OCH2OCH3', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
               'CH4', 'CH2O', 'CH3OH', 'HO2', 'CH3O', 'HCO']
SPECIES_RANGES = {'CH3OCH2OCH3': (0.01, 0.05), 'O2': (0.10, 0.25),
                  'N2': (0.60, 0.80), 'CO2': (0.00, 0.005), 'H2O': (0.00, 0.02)}
BATCHES = [
    ('train_dmm',          15, [800, 1600],  [0.0, 1.3], None),
    ('train_dmm_expanded', 60, [700, 1600],  [0.0, 1.3],
     [(700, 900, 24), (900, 1100, 16), (1100, 1600, 20)]),
    ('train_dmm_hiT',      30, [1200, 1600], [0.0, 1.0], None),
    ('train_dmm_midT',     15, [950, 1150],  [0.0, 0.5], None),
]
ENSEMBLE_SEEDS = 12
BALANCED_DEFAULT = 'train_dmm_scratch_balanced'

def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
def run(cmd):
    log('$ ' + ' '.join(cmd))
    r = subprocess.run(cmd)


def step1_generate():
    from slgps.make_data_parallel import make_data_parallel
    for name, n, t_rng, p_rng, strata in BATCHES:
        path = os.path.join('data', name)
        if os.path.isfile(os.path.join(path, 'data.csv')):
            log(f'STEP 1: {name} exists, skipping'); continue
        os.makedirs(path, exist_ok=True)
        log(f'STEP 1: generating {name} ({n} cases)')
        kwargs = dict(fuel=FUEL, mech_file=MECH,
            end_threshold=2e5, ign_HRR_threshold_div=300,
            ign_GPS_resolution=100, norm_GPS_resolution=20, GPS_per_interval=2,
            n_cases=n, t_rng=t_rng, p_rng=p_rng, phi_rng=[0.7, 1.4],
            alpha=0.001, always_threshold=0.99, never_threshold=0.01,
            pathname=path, species_ranges=SPECIES_RANGES)
        try:
            if strata: make_data_parallel(**kwargs, t_strata=strata)
            else: make_data_parallel(**kwargs)
        except TypeError:
            log('  (t_strata unsupported; uniform sampling)')
            make_data_parallel(**kwargs)

def step2_balance(bal):
    if os.path.isfile(os.path.join('data', bal, 'data.csv')):
        log(f'STEP 2: {bal} exists, skipping'); return
    run([sys.executable, 'make_balanced_data.py', bal, '500'] +
        [b[0] for b in BATCHES])

def step3_audit(bal):
    log('STEP 3: never-list audit')
    run([sys.executable, 'audit_never_list.py', os.path.join('data', bal)])

def step4_train(bal, tag):
    import numpy as np, tensorflow as tf
    from pandas import read_csv
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from joblib import Parallel, delayed
    from pickle import dump
    mdir = os.path.join('models', tag)
    mp = os.path.join(mdir, 'model.h5')
    if os.path.isfile(mp):
        log(f'STEP 4: {mp} exists, skipping'); return
    log(f'STEP 4: training ensemble ({ENSEMBLE_SEEDS} seeds)')
    def spec_train(X, Y, seed):
        tf.keras.utils.set_random_seed(seed)
        Xtr, Xv, Ytr, Yv = train_test_split(X, Y, test_size=0.2, random_state=seed)
        m = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(Y.shape[1], activation='sigmoid')])
        m.compile(optimizer='adam', loss='binary_crossentropy')
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                              patience=30, restore_best_weights=True)
        m.fit(Xtr, Ytr, epochs=200, batch_size=32, callbacks=[es],
              verbose=0, validation_data=(Xv, Yv))
        return m, m.evaluate(Xv, Yv, verbose=0)
    dd = os.path.join('data', bal)
    X = read_csv(os.path.join(dd, 'data.csv'))
    Y = read_csv(os.path.join(dd, 'species.csv')).iloc[:, :-1]
    X = X[['# Temperature', 'Atmospheres'] + INPUT_SPECS]
    scaler = preprocessing.MinMaxScaler()
    Xp = scaler.fit_transform(X)
    results = Parallel(n_jobs=min(ENSEMBLE_SEEDS, os.cpu_count() or 1))(
        delayed(spec_train)(Xp, Y.to_numpy(), s) for s in range(ENSEMBLE_SEEDS))
    vals = [v for _, v in results]
    best = int(np.argmin(vals))
    log(f'  best seed {best}, val_loss={vals[best]:.4f}')
    os.makedirs(mdir, exist_ok=True)
    results[best][0].save(mp, save_format='h5')
    with open(os.path.join(mdir, 'scaler.pkl'), 'wb') as f: dump(scaler, f)

def step5_validate(bal, tag):
    ed = os.path.join('data', f'train_{tag}')
    if not os.path.isdir(ed):
        import shutil; shutil.copytree(os.path.join('data', bal), ed)
    run([sys.executable, 'validate_full.py', tag])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='dmm_scratch')
    args = ap.parse_args()
    t0 = time.time()
    step1_generate()
    step2_balance(BALANCED_DEFAULT)
    step3_audit(BALANCED_DEFAULT)
    step4_train(BALANCED_DEFAULT, args.tag)
    step5_validate(BALANCED_DEFAULT, args.tag)
    log(f'PIPELINE COMPLETE ({(time.time() - t0) / 3600:.1f} h)')

    if r.returncode != 0: sys.exit(f'FAILED (exit {r.returncode})')
