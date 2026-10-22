"""
Never-list audit: verifies the ANN's reachable species set is sufficient.

Usage: python3 audit_never_list.py [data_dir]
"""
import os, sys
import numpy as np
import pandas as pd
import cantera as ct

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, '..', '..', 'src'))

from slgps.GPS.src.core.def_GPS import GPS_algo
from slgps.GPS.src.core.def_build_graph import build_flux_graph
from slgps.utils import auto_ign_build_X0, sub_mech

MECH = os.path.join(BASE, 'mechanism', 'PODE-1_clean.yaml')
FUEL = 'CH3OCH2OCH3'
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(BASE, 'data', 'train_dmm_balanced')
ELEMENTS = ['C', 'H', 'O']
SOURCES = [FUEL, 'O2']
TARGETS = ['CO2', 'H2O']
ALPHA = 0.001
N_SAMPLES = 20
TOL_PCT = 5.0

CASES = [
    ('T1500_P1',      1500,  1.0, 1.0),
    ('T1200_P1',      1200,  1.0, 1.0),
    ('T1000_P1',      1000,  1.0, 1.0),
    ('T1100_P1_rich', 1100,  1.0, 1.3),
    ('T900_P10',       900, 10.0, 1.0),
    ('T800_P20',       800, 20.0, 1.0),
    ('T750_P20',       750, 20.0, 1.0),
    ('T1400_P5_lean', 1400,  5.0, 0.7),
    ('T750_P10',       750, 10.0, 1.0),
    ('T1000_P10',     1000, 10.0, 1.0),
    ('T1600_P1',      1600,  1.0, 1.0),
    ('X_T700_P20',     700, 20.0, 1.0),
    ('X_T900_P30',     900, 30.0, 1.0),
    ('X_T1100_P5_vlean', 1100, 5.0, 0.5),
    ('X_T1100_P5_vrich', 1100, 5.0, 1.6),
]

def load_names(fname):
    n = pd.read_csv(os.path.join(DATA_DIR, fname)).columns.to_list()[:-1]
    if n and n[0][:2] == '# ': n[0] = n[0][2:]
    return n

def idt(raw):
    t = np.asarray(raw['axis0']); T = np.asarray(raw['temperature'])
    return t[np.argmax(np.diff(T) / np.diff(t))]

never = set(load_names('never_spec_nums.csv'))
var = set(load_names('var_spec_nums.csv'))
always = set(load_names('always_spec_nums.csv'))
ceiling_species = var | always
print(f"partition: var={len(var)} always={len(always)} never={len(never)}")

failures = {}
for label, T0, P, phi in CASES:
    soln = ct.Solution(MECH)
    soln.TP = T0, P * 101325
    soln.set_equivalence_ratio(phi, FUEL + ':1.0', 'O2:1.0, N2:3.76')
    X0 = soln.mole_fraction_dict()
    det = ct.Solution(MECH)
    raw, _, _ = auto_ign_build_X0(det, T0, P, X0, end_threshold=2e3,
                                   end=2.0, dir_raw='ign')
    N = len(raw['axis0'])
    idt_det = idt(raw)

    required = set(SOURCES) | set(TARGETS) | {'N2'}
    s = ct.Solution(MECH)
    for i in sorted(set(np.linspace(1, N - 2, N_SAMPLES).astype(int))):
        for e in ELEMENTS:
            fg = build_flux_graph(s, raw, e, path_save='flux_graph_tmp',
                                  overwrite=True, i0=int(i), i1='eq', constV=False)
            for sc in SOURCES:
                for tg in TARGETS:
                    if sc in fg.nodes() and tg in fg.nodes():
                        r = GPS_algo(s, fg, sc, tg, path_save=None, K=1,
                                     alpha=ALPHA, beta=0.5, normal='max',
                                     iso=None, overwrite=False, raw='unknown',
                                     notes=None, gamma=None)
                        required |= set(r['species'].keys())

    blocked = sorted(required & never)
    if not blocked:
        print(f"  {label:16s} GPS-required={len(required):3d}  blocked=0  ok")
        continue

    ceil = sub_mech(MECH, ceiling_species)
    X0c = {k: v for k, v in X0.items() if k in set(ceil.species_names)}
    raw_c, _, _ = auto_ign_build_X0(ceil, T0, P, X0c, end_threshold=2e3,
                                     end=2.0, dir_raw='ign')
    err = 100 * abs(idt(raw_c) - idt_det) / idt_det
    status = 'FAIL' if err > TOL_PCT else 'ok(ceiling)'
    print(f"  {label:16s} GPS-required={len(required):3d}  blocked={len(blocked):2d}  "
          f"ceiling-idt-err={err:5.2f}%  {status}")
    if err > TOL_PCT:
        failures[label] = (blocked, err)

print()
if failures:
    print(f"AUDIT FAILED: never list degrades ceiling ignition beyond {TOL_PCT}%")
    sys.exit(1)
print("AUDIT PASSED: never list does not harm ignition at any audited condition.")
