"""
Validate the merged-data model on key cases:
  - T1500_P1: high-T regression case (was 146.7% with expanded model)
  - T1200_P1, T1000_P1, T1100_P1_rich: marginal ~15% cases
  - T900_P10, T800_P20: NTC guards (must stay <1%)

Uses local src/slgps (evolving-state fix ported). Results -> results/ch3ocho_merged/.
"""
import os
import sys
import time
import pickle
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, '..', '..', 'src'))

import cantera as ct
from slgps.utils import auto_ign_build_SL, auto_ign_build_X0

TAG = sys.argv[1] if len(sys.argv) > 1 else 'ch3ocho_merged'
MECH = os.path.join(BASE, 'ThInK_1.0_Mech.yaml')
FUEL = 'CH3OCHO'
DATA_DIR = os.path.join(BASE, 'data', f'train_{TAG}')
MODEL = os.path.join(BASE, 'models', TAG, 'model.h5')
SCALER = os.path.join(BASE, 'models', TAG, 'scaler.pkl')
RESULTS = os.path.join(BASE, 'results', f'{TAG}_idtscaled')
os.makedirs(RESULTS, exist_ok=True)
print(f'model tag: {TAG}\n  data: {DATA_DIR}\n  results: {RESULTS}')

INPUT_SPECS = ['CH3OCHO', 'O2', 'H2O', 'CO2', 'CO', 'OH', 'H', 'H2',
               'CH4', 'CH2O', 'CH3OH', 'HO2', 'CH3O', 'HCO']

CASES = [
    ('CH3OCHO_T1500_P1',      1500,  1.0, 1.0, 0.005),
    ('CH3OCHO_T1200_P1',      1200,  1.0, 1.0, 0.02),
    ('CH3OCHO_T1000_P1',      1000,  1.0, 1.0, 0.2),
    ('CH3OCHO_T1100_P1_rich', 1100,  1.0, 1.3, 0.05),
    ('CH3OCHO_T900_P10',       900, 10.0, 1.0, 0.1),
    ('CH3OCHO_T800_P20',       800, 20.0, 1.0, 0.5),
]


def ign_delay(result):
    if result is None:
        return None
    t = np.array(result[0]['axis0'])
    T = np.array(result[0]['temperature'])
    if len(t) < 3:
        return None
    dTdt = np.diff(T) / np.diff(t)
    return t[np.argmax(dTdt)]


summary = []
for label, T0, P_atm, phi, t_end in CASES:
    out = os.path.join(RESULTS, f'{label}.pkl')
    print(f'\n=== {label}  T={T0}K P={P_atm}atm phi={phi} ===', flush=True)

    case = {'config': dict(label=label, T0=T0, P_atm=P_atm, phi=phi, t_end=t_end)}

    t0 = time.time()
    soln = ct.Solution(MECH)
    soln.TP = T0, P_atm * 101325
    soln.set_equivalence_ratio(phi, f'{FUEL}:1.0', 'O2:1.0, N2:3.76')
    X0 = soln.mole_fraction_dict()
    case['detailed'] = auto_ign_build_X0(soln, T0, P_atm, X0, end_threshold=None, end=t_end, dir_raw='ign')
    print(f'  detailed done ({time.time()-t0:.0f}s, {len(case["detailed"][0]["axis0"])} steps)', flush=True)

    # scale ANN re-prediction intervals to this case's own ignition delay so every
    # case gets >=50 mechanism updates during induction (no hand-tuned T/P branches)
    idt_det = ign_delay(case['detailed'])
    norm_Dt = min(2e-4, idt_det / 50)
    ign_Dt = min(idt_det / 200, norm_Dt / 4)  # ignition step must stay finer than induction step
    print(f'  timesteps from idt={idt_det:.3e}: norm_Dt={norm_Dt:.2e} ign_Dt={ign_Dt:.2e}', flush=True)

    t0 = time.time()
    case['slgps'] = auto_ign_build_SL(FUEL, MECH, INPUT_SPECS, norm_Dt, ign_Dt,
                                      T0, phi, P_atm, t_end, SCALER, MODEL, DATA_DIR, 9e7)
    print(f'  slgps done ({time.time()-t0:.0f}s)', flush=True)

    with open(out, 'wb') as f:
        pickle.dump(case, f)

    d, s = ign_delay(case['detailed']), ign_delay(case['slgps'])
    err = 100 * abs(s - d) / d if (d and s) else float('nan')
    nspec = np.mean(case['slgps'][3]) if case['slgps'] else float('nan')
    summary.append((label, T0, P_atm, phi, d, s, err, nspec))
    print(f'  idt: det={d:.4e}  sl={s:.4e}  err={err:.1f}%  avg_species={nspec:.0f}', flush=True)

print('\n' + '=' * 80)
print(f'{"case":26s} {"T":>5} {"P":>5} {"phi":>4} {"det idt":>11} {"sl idt":>11} {"err%":>7} {"nsp":>4}')
for label, T0, P, phi, d, s, err, nsp in summary:
    mark = 'PASS' if err < 10 else ('WARN' if err < 25 else 'FAIL')
    print(f'{label:26s} {T0:>5.0f} {P:>5.1f} {phi:>4.1f} {d:>11.4e} {s:>11.4e} {err:>6.1f}% {nsp:>4.0f} {mark}')
