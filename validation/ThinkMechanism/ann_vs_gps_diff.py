"""
ANN-vs-GPS species diff along a detailed autoignition trajectory.

For each sampled state on the DETAILED trajectory:
  - ANN set: always_specs + var_specs with sigmoid > 0.5, using the SAME
    evolving-state input construction as auto_ign_build_SL (T, P, input_specs X).
  - GPS set: classic GPS species union at that state (all elements/sources/targets).

Reports, per sample, GPS-required species the ANN misses, split into:
  - in 'never' list (ANN can never include them)  <- structural failure
  - in 'var' list but predicted < 0.5             <- prediction failure

Usage: python3 ann_vs_gps_diff.py [T0] [P_atm] [phi] [n_samples]
Defaults: 1500 1 1.0 12
"""
import os
import sys
import numpy as np
import pandas as pd
import cantera as ct
from pickle import load as pkl_load

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from tensorflow.keras.models import load_model

from slgps.GPS.src.core.def_GPS import GPS_algo
from slgps.GPS.src.core.def_build_graph import build_flux_graph
from slgps.utils import auto_ign_build_X0

BASE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.join(BASE, 'ThInK_1.0_Mech.yaml')
FUEL = 'CH3OCHO'
DATA_DIR = os.path.join(BASE, 'data', 'train_ch3ocho_merged')
MODEL = os.path.join(BASE, 'models', 'ch3ocho_merged', 'model.h5')
SCALER = os.path.join(BASE, 'models', 'ch3ocho_merged', 'scaler.pkl')

INPUT_SPECS = ['CH3OCHO', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
               'CH4', 'CH2O', 'CH3OH', 'HO2', 'CH3O', 'HCO']
ELEMENTS = ['C', 'H', 'O']
SOURCES = [FUEL, 'O2']
TARGETS = ['CO2', 'H2O']
ALPHA = 0.001

T0 = float(sys.argv[1]) if len(sys.argv) > 1 else 1500.0
P_ATM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
PHI = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
N_SAMPLES = int(sys.argv[4]) if len(sys.argv) > 4 else 12


def load_spec_list(fname):
    specs = pd.read_csv(os.path.join(DATA_DIR, fname)).columns.to_list()[:-1]
    if specs and specs[0][:2] == '# ':
        specs[0] = specs[0][2:]
    return specs


var_specs = load_spec_list('var_spec_nums.csv')
always_specs = load_spec_list('always_spec_nums.csv')
never_specs = load_spec_list('never_spec_nums.csv')
print(f"species partition: var={len(var_specs)} always={len(always_specs)} never={len(never_specs)}")

model = load_model(MODEL)
scaler = pkl_load(open(SCALER, 'rb'))

# --- detailed trajectory -----------------------------------------------------
soln = ct.Solution(MECH)
soln.TP = T0, P_ATM * 101325
soln.set_equivalence_ratio(PHI, FUEL + ':1.0', 'O2:1.0, N2:3.76')
X0 = soln.mole_fraction_dict()
det = ct.Solution(MECH)
raw, _, _ = auto_ign_build_X0(det, T0, P_ATM, X0, end_threshold=2e3, end=2.0, dir_raw='ign')
t = np.asarray(raw['axis0'])
T = np.asarray(raw['temperature'])
mf = np.asarray(raw['mole_fraction'])
spec_names = ct.Solution(MECH).species_names
idt = t[np.argmax(np.diff(T) / np.diff(t))]
print(f"detailed: steps={len(t)} Tmax={T.max():.0f} idt={idt:.4e}s")

# sample most densely around ignition
idx = sorted(set(np.linspace(1, len(t) - 2, N_SAMPLES).astype(int)))


def ann_set(i):
    """ANN prediction using evolving detailed state at index i."""
    x = dict(zip(spec_names, mf[i]))
    vec = [T[i], P_ATM] + [x.get(s, 0.0) for s in INPUT_SPECS]
    pred = model.predict(scaler.transform(np.array([vec])), verbose=0)[0]
    kept = set(always_specs)
    kept |= {var_specs[k] for k in range(len(pred)) if pred[k] > 0.5}
    return kept, pred


def gps_set(i):
    s = ct.Solution(MECH)
    kept = set(SOURCES) | set(TARGETS) | {'N2'}
    for e in ELEMENTS:
        fg = build_flux_graph(s, raw, e, path_save='flux_graph_tmp', overwrite=True,
                              i0=int(i), i1='eq', constV=False)
        for sc in SOURCES:
            for tg in TARGETS:
                if sc in fg.nodes() and tg in fg.nodes():
                    r = GPS_algo(s, fg, sc, tg, path_save=None, K=1, alpha=ALPHA,
                                 beta=0.5, normal='max', iso=None, overwrite=False,
                                 raw='unknown', notes=None, gamma=None)
                    kept |= set(r['species'].keys())
    return kept


print(f"\n{'idx':>6} {'t/idt':>7} {'T(K)':>7} {'ANN':>4} {'GPS':>4} "
      f"{'miss':>4} {'never':>5}  missing-species (n=never, v=var<0.5)")
print('-' * 110)

union_missing_never, union_missing_var = {}, {}
for i in idx:
    a, pred = ann_set(i)
    g = gps_set(i)
    missing = g - a
    m_never = sorted(sp for sp in missing if sp in never_specs)
    m_var = sorted(sp for sp in missing if sp in var_specs)
    m_other = sorted(sp for sp in missing if sp not in never_specs and sp not in var_specs)
    for sp in m_never:
        union_missing_never[sp] = union_missing_never.get(sp, 0) + 1
    for sp in m_var:
        union_missing_var[sp] = union_missing_var.get(sp, 0) + 1
    tags = [f"{sp}(n)" for sp in m_never] + [f"{sp}(v)" for sp in m_var] + \
           [f"{sp}(?)" for sp in m_other]
    print(f"{i:>6} {t[i]/idt:>7.3f} {T[i]:>7.1f} {len(a):>4} {len(g):>4} "
          f"{len(missing):>4} {len(m_never):>5}  {', '.join(tags)}")

print("\n=== aggregate: GPS-required species the ANN missed (count over samples) ===")
print(" stuck in NEVER list (ANN cannot include):")
for sp, c in sorted(union_missing_never.items(), key=lambda kv: -kv[1]):
    print(f"   {sp:12s} {c}/{len(idx)}")
print(" in VAR list but predicted <0.5:")
for sp, c in sorted(union_missing_var.items(), key=lambda kv: -kv[1]):
    print(f"   {sp:12s} {c}/{len(idx)}")
