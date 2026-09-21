"""Plot reduced-mechanism species count vs time for representative cases.

Reads validation pickles: case['slgps'] = (result, mech_times, n_rxns, n_species).
Time normalized by detailed ignition delay to overlay cases of different idt.
"""
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'results', 'ch3ocho_balanced2v_idtscaled')

CASES = ['CH3OCHO_T750_P20', 'CH3OCHO_T900_P10', 'CH3OCHO_T1200_P10', 'CH3OCHO_T1500_P1']
COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']


def idt(r):
    t = np.asarray(r[0]['axis0'], float)
    T = np.asarray(r[0]['temperature'], float)
    return t[np.argmax(np.diff(T) / np.diff(t))]


fig, ax = plt.subplots(figsize=(10, 5.5))
for label, color in zip(CASES, COLORS):
    fp = os.path.join(RESULTS_DIR, f'{label}.pkl')
    with open(fp, 'rb') as f:
        case = pickle.load(f)
    det, sl = case['detailed'], case['slgps']
    tau = idt(det)
    mech_times = np.asarray(sl[1], float) / tau
    n_spec = np.asarray(sl[3], float)
    ax.step(mech_times, n_spec, where='post', color=color, lw=1.6,
            label=f"{label.replace('CH3OCHO_', '')} (mean {n_spec.mean():.0f})")

ax.axhline(160, color='k', ls='--', lw=1, label='Detailed (160)')
ax.axvline(1.0, color='0.5', ls=':', lw=1)
ax.text(1.02, 150, 'ignition', fontsize=9, color='0.4')
ax.set_xlim(0, 2.0)
ax.set_xlabel(r'Normalized time $t/\tau_{ign}$')
ax.set_ylabel('Active species in reduced mechanism')
ax.set_title('ThInK 1.0 / CH$_3$OCHO — adaptive mechanism size vs time (final model)')
ax.legend(loc='center right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'CH3OCHO_nspecies_vs_time.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'saved {out}')
