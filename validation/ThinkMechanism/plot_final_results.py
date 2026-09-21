"""Generate per-case comparison plots from final-model validation pickles.

Cantera-free (species order read from training data.csv header) so it can run
on the login node while validation jobs are active.

Usage: python3 plot_final_results.py [results_dir] [data_dir]
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
DATA_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'data', 'train_ch3ocho_balanced2')

with open(os.path.join(DATA_DIR, 'data.csv')) as f:
    header = f.readline().lstrip('# ').strip().split(',')
species_names = header[2:]

PLOT_SPECS = ['CH3OCHO', 'O2', 'CO2', 'H2O', 'CO']


def idt(r):
    t = np.asarray(r[0]['axis0'], float)
    T = np.asarray(r[0]['temperature'], float)
    return t[np.argmax(np.diff(T) / np.diff(t))]


for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith('.pkl'):
        continue
    fpath = os.path.join(RESULTS_DIR, fname)
    try:
        with open(fpath, 'rb') as f:
            case = pickle.load(f)
    except Exception as e:
        print(f'skip {fname}: {e}')  # possibly still being written
        continue
    label = case['config']['label']
    det, sl = case.get('detailed'), case.get('slgps')
    if det is None or sl is None:
        print(f'skip {label}: missing runs')
        continue

    err = 100 * abs(idt(sl) - idt(det)) / idt(det)
    cfg = case['config']
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"ThInK 1.0 — {label} (final model)\n"
                 f"T0={cfg['T0']}K, P={cfg['P_atm']}atm, phi={cfg['phi']}  |  idt err {err:.1f}%",
                 fontsize=11, fontweight='bold')

    t_d = np.asarray(det[0]['axis0'], float) * 1000
    t_s = np.asarray(sl[0]['axis0'], float) * 1000
    axs[0].plot(t_d, det[0]['temperature'], 'k-', lw=1.5, label='Detailed')
    axs[0].plot(t_s, sl[0]['temperature'], 'r--', lw=1.5, label='SL-GPS')
    axs[0].set_ylabel('Temperature (K)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_d, det[0]['heat_release_rate'], 'k-', lw=1.5)
    axs[1].plot(t_s, sl[0]['heat_release_rate'], 'r--', lw=1.5)
    axs[1].set_ylabel('HRR (W/m³)')
    axs[1].set_yscale('symlog', linthresh=1e4)
    axs[1].grid(True, alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, len(PLOT_SPECS)))
    mf_d = np.asarray(det[0]['mole_fraction'], float)
    mf_s = np.asarray(sl[0]['mole_fraction'], float)
    for j, spec in enumerate(PLOT_SPECS):
        if spec in species_names:
            idx = species_names.index(spec)
            if mf_d.ndim == 2 and idx < mf_d.shape[1]:
                axs[2].plot(t_d, mf_d[:, idx], '-', color=colors[j], lw=1.2, label=spec)
            if mf_s.ndim == 2 and idx < mf_s.shape[1]:
                axs[2].plot(t_s, mf_s[:, idx], '--', color=colors[j], lw=1.0)
    axs[2].set_ylabel('Mole Fraction (solid=det, dashed=SL)')
    axs[2].set_xlabel('Time (ms)')
    axs[2].legend(loc='best', fontsize=9)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f'{label}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'saved {label}.png  (err {err:.1f}%)')
