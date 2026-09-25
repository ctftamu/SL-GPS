"""
End-to-end from-scratch reproduction of the ThInK 1.0 / CH3OCHO SL-GPS study
(report Revision 2, release v2.1.x).

Stages (each skipped if its output already exists, so the run is resumable):
  1. Generate the four training batches (make_data_parallel)
  2. Build the temperature-balanced training set (make_balanced_data.py)
  3. Never-list audit (audit_never_list.py) -- aborts on failure
  4. Train the ANN ensemble, selecting by VALIDATION loss
  5. Full 35-case validation (validate_full.py)

Usage:
    python recreate_from_scratch.py            # full pipeline
    python recreate_from_scratch.py --tag NAME # custom model tag (default ch3ocho_scratch)

Note: data generation is randomized (unseeded initial conditions), so per-case
errors will differ slightly from the report tables. Expected outcome: all 31
in-envelope cases pass (<10% idt error), NTC region sub-1%, and exactly one
failure at the pressure-extrapolation probe X_T900_P30.
"""
import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(BASE, '..', '..', 'src'))
sys.path.insert(0, SRC)
os.chdir(BASE)

MECH = 'ThInK_1.0_Mech.yaml'
FUEL = 'CH3OCHO'
INPUT_SPECS = ['CH3OCHO', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
               'CH4', 'CH2O', 'CH3OH', 'HO2', 'CH3O', 'HCO']
SPECIES_RANGES = {'CH3OCHO': (0.01, 0.05), 'O2': (0.10, 0.25),
                  'N2': (0.60, 0.80), 'CO2': (0.00, 0.005), 'H2O': (0.00, 0.02)}

# (dir, n_cases, t_rng, p_rng, t_strata or None)
BATCHES = [
    ('train_ch3ocho',          15, [800, 1600],  [0.0, 1.3], None),
    ('train_ch3ocho_expanded', 60, [700, 1600],  [0.0, 1.3],
     [(700, 900, 24), (900, 1100, 16), (1100, 1600, 20)]),
    ('train_ch3ocho_hiT',      30, [1200, 1600], [0.0, 1.0], None),
    ('train_ch3ocho_midT',     15, [950, 1150],  [0.0, 0.5], None),
]

ENSEMBLE_SEEDS = 12
BALANCED_NAME_DEFAULT = 'train_ch3ocho_scratch_balanced'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run(cmd):
    log('$ ' + ' '.join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f'FAILED (exit {r.returncode}): {" ".join(cmd)}')


def step1_generate():
    from slgps.make_data_parallel import make_data_parallel
    for name, n, t_rng, p_rng, strata in BATCHES:
        path = os.path.join('data', name)
        if os.path.isfile(os.path.join(path, 'data.csv')):
            log(f'STEP 1: {name} exists, skipping')
            continue
        os.makedirs(path, exist_ok=True)  # make_data_parallel crashes without it
        log(f'STEP 1: generating {name} ({n} cases, T {t_rng}, logP {p_rng})')
        kwargs = dict(
            fuel=FUEL, mech_file=MECH,
            end_threshold=2e5, ign_HRR_threshold_div=300,
            ign_GPS_resolution=100, norm_GPS_resolution=20, GPS_per_interval=2,
            n_cases=n, t_rng=t_rng, p_rng=p_rng, phi_rng=[0.7, 1.4],
            alpha=0.001, always_threshold=0.99, never_threshold=0.01,
            pathname=path, species_ranges=SPECIES_RANGES,
        )
        try:
            if strata:
                make_data_parallel(**kwargs, t_strata=strata)
            else:
                make_data_parallel(**kwargs)
        except TypeError:
            # older make_data_parallel without t_strata: plain uniform sampling
            log('  (t_strata unsupported by this slgps version; using uniform t_rng)')
            make_data_parallel(**kwargs)


def step2_balance(balanced_name):
    if os.path.isfile(os.path.join('data', balanced_name, 'data.csv')):
        log(f'STEP 2: {balanced_name} exists, skipping')
        return
    run([sys.executable, 'make_balanced_data.py', balanced_name, '500'] +
        [b[0] for b in BATCHES])


def step3_audit(balanced_name):
    log('STEP 3: never-list audit (aborts on failure)')
    run([sys.executable, 'audit_never_list.py', os.path.join('data', balanced_name)])


def step4_train(balanced_name, tag):
    mdir = os.path.join('models', tag)
    model_path = os.path.join(mdir, 'model.h5')
    if os.path.isfile(model_path):
        log(f'STEP 4: {model_path} exists, skipping')
        return
    log(f'STEP 4: training ensemble ({ENSEMBLE_SEEDS} seeds, val-loss selection)')
    import numpy as np
    import tensorflow as tf
    from pandas import read_csv
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from joblib import Parallel, delayed
    from pickle import dump

    def spec_train(X, Y, seed):
        tf.keras.utils.set_random_seed(seed)
        Xtr, Xval, Ytr, Yval = train_test_split(X, Y, test_size=0.2, random_state=seed)
        m = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(Y.shape[1], activation='sigmoid')])
        m.compile(optimizer='adam', loss='binary_crossentropy')
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                              patience=30, restore_best_weights=True)
        m.fit(Xtr, Ytr, epochs=200, batch_size=32, callbacks=[es],
              verbose=0, validation_data=(Xval, Yval))
        return m, m.evaluate(Xval, Yval, verbose=0)

    data_dir = os.path.join('data', balanced_name)
    X = read_csv(os.path.join(data_dir, 'data.csv'))
    Y = read_csv(os.path.join(data_dir, 'species.csv')).iloc[:, :-1]
    X = X[['# Temperature', 'Atmospheres'] + INPUT_SPECS]
    scaler = preprocessing.MinMaxScaler()
    Xp = scaler.fit_transform(X)
    results = Parallel(n_jobs=min(ENSEMBLE_SEEDS, os.cpu_count() or 1))(
        delayed(spec_train)(Xp, Y.to_numpy(), s) for s in range(ENSEMBLE_SEEDS))
    vals = [v for _, v in results]
    best = int(np.argmin(vals))
    log(f'  val losses: {["%.4f" % v for v in vals]} -> best seed {best}')
    os.makedirs(mdir, exist_ok=True)
    results[best][0].save(model_path, save_format='h5')
    with open(os.path.join(mdir, 'scaler.pkl'), 'wb') as f:
        dump(scaler, f)
    log(f'  saved {model_path}')


def step5_validate(balanced_name, tag):
    # validate_full.py resolves data dir as data/train_<tag>: provide a symlink-free copy check
    expected_data = os.path.join('data', f'train_{tag}')
    if not os.path.isdir(expected_data):
        import shutil
        shutil.copytree(os.path.join('data', balanced_name), expected_data)
    run([sys.executable, 'validate_full.py', tag])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='ch3ocho_scratch')
    args = ap.parse_args()
    t0 = time.time()
    step1_generate()
    step2_balance(BALANCED_NAME_DEFAULT)
    step3_audit(BALANCED_NAME_DEFAULT)
    step4_train(BALANCED_NAME_DEFAULT, args.tag)
    step5_validate(BALANCED_NAME_DEFAULT, args.tag)
    log(f'PIPELINE COMPLETE ({(time.time() - t0) / 3600:.1f} h)')
