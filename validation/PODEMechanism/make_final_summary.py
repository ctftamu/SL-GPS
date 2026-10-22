"""Build the summary figure (idt comparison + mechanism size) from validation results."""
import numpy as np
import os, sys, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(BASE, 'results', 'dmm_balanced_idtscaled')
FULL_N_SPEC = 586


def idt(r):
    t = np.asarray(r[0]['axis0'], float)
    T = np.asarray(r[0]['temperature'], float)
    return t[np.argmax(np.diff(T) / np.diff(t))]


labels, det_vals, sl_vals, errs, nsps = [], [], [], [], []
for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith('.pkl'):
        continue
    with open(os.path.join(RESULTS_DIR, fname), 'rb') as f:
        case = pickle.load(f)
    det, sl = case.get('detailed'), case.get('slgps')
    if det is None or sl is None:
        continue
    d, s = idt(det), idt(sl)
    err = 100 * abs(s - d) / d
    nsp = np.mean(sl[3])
    label = case['config']['label'].replace('DMM_', '')
    labels.append(label)
    det_vals.append(d * 1000)
    sl_vals.append(s * 1000)
    errs.append(err)
    nsps.append(nsp)

det_arr = np.array(det_vals)
sl_arr = np.array(sl_vals)
err_arr = np.array(errs)
nsp_arr = np.array(nsps)
x = np.arange(len(labels))

n_pass = sum(1 for e in errs if e < 10)
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle(f'PODE-1 / DMM — Final model: {n_pass}/{len(labels)} cases pass, '
             f'mean err {err_arr.mean():.1f}%',
             fontsize=12, fontweight='bold')

w = 0.38
axs[0].bar(x - w/2, det_arr, w, label='Detailed', color='0.25')
axs[0].bar(x + w/2, sl_arr, w, label='SL-GPS', color='tab:red', alpha=0.85)
for i, e in enumerate(errs):
    axs[0].text(i, max(det_arr[i], sl_arr[i])*1.25, f'{e:.1f}%',
                ha='center', fontsize=7)
axs[0].set_yscale('log')
axs[0].set_ylabel('Ignition delay (ms)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
axs[0].legend()
axs[0].grid(True, axis='y', alpha=0.3)
axs[0].set_title('Ignition delay: detailed vs SL-GPS (label = error %)')

axs[1].bar(x, nsp_arr, color='tab:blue', alpha=0.85)
axs[1].axhline(FULL_N_SPEC, color='k', ls='--', lw=1,
               label=f'Detailed ({FULL_N_SPEC} species)')
axs[1].set_ylabel('Mean active species')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
axs[1].legend()
axs[1].grid(True, axis='y', alpha=0.3)
axs[1].set_title(f'Mechanism size (mean {nsp_arr.mean():.0f}/{FULL_N_SPEC} '
                 f'= {100*(1-nsp_arr.mean()/FULL_N_SPEC):.0f}% reduction)')

plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'DMM_summary.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f'saved {out}')
