"""Build the final-model summary figure (idt comparison + mechanism size)
from the 17-case validation results (job 6425, model ch3ocho_balanced2v)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# label, det_idt(s), sl_idt(s), err%, avg_species  (slurm_final2_6425.log, B table)
RESULTS = [
    ('T750_P20',       8.3133e-2, 8.2993e-2, 0.2, 35),
    ('T800_P5',        1.2080e-1, 1.1989e-1, 0.8, 30),
    ('T800_P20',       2.5512e-2, 2.5406e-2, 0.4, 33),
    ('T850_P10',       2.5596e-2, 2.5387e-2, 0.8, 31),
    ('T900_P5',        3.3776e-2, 3.3695e-2, 0.2, 30),
    ('T900_P10',       1.3703e-2, 1.3602e-2, 0.7, 31),
    ('T900_P15',       8.1433e-3, 8.0862e-3, 0.7, 29),
    ('T950_P10',       7.1282e-3, 7.0819e-3, 0.6, 29),
    ('T1000_P1',       4.1210e-2, 4.3051e-2, 4.5, 23),
    ('T1050? T1100_P5', 1.8704e-3, 1.9042e-3, 1.8, 21),
    ('T1100_P1_rich',  7.9617e-3, 8.4722e-3, 6.4, 21),
    ('T1100_P10_rich', 8.2455e-4, 8.2767e-4, 0.4, 24),
    ('T1200_P1',       2.6577e-3, 2.8897e-3, 8.7, 21),
    ('T1200_P10',      2.8001e-4, 2.7926e-4, 0.3, 26),
    ('T1300_P2',       4.7965e-4, 4.9938e-4, 4.1, 19),
    ('T1400_P5_lean',  9.3681e-5, 9.6363e-5, 2.9, 19),
    ('T1500_P1',       2.2565e-4, 2.4343e-4, 7.9, 19),
]
RESULTS[9] = ('T1100_P5', 1.8704e-3, 1.9042e-3, 1.8, 21)

labels = [r[0] for r in RESULTS]
det = np.array([r[1] for r in RESULTS]) * 1000
sl = np.array([r[2] for r in RESULTS]) * 1000
err = np.array([r[3] for r in RESULTS])
nsp = np.array([r[4] for r in RESULTS])
x = np.arange(len(labels))

fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle('ThInK 1.0 / CH$_3$OCHO — Final model (v2.1.0): 17/17 cases pass, mean err 2.4%',
             fontsize=12, fontweight='bold')

w = 0.38
axs[0].bar(x - w/2, det, w, label='Detailed', color='0.25')
axs[0].bar(x + w/2, sl, w, label='SL-GPS', color='tab:red', alpha=0.85)
for i, e in enumerate(err):
    axs[0].text(i, max(det[i], sl[i]) * 1.25, f'{e:.1f}%', ha='center', fontsize=7)
axs[0].set_yscale('log')
axs[0].set_ylabel('Ignition delay (ms)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
axs[0].legend()
axs[0].grid(True, axis='y', alpha=0.3)
axs[0].set_title('Ignition delay: detailed vs SL-GPS (label = error %)')

axs[1].bar(x, nsp, color='tab:blue', alpha=0.85)
axs[1].axhline(160, color='k', ls='--', lw=1, label='Detailed (160 species)')
axs[1].set_ylabel('Mean active species')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
axs[1].legend()
axs[1].grid(True, axis='y', alpha=0.3)
axs[1].set_title(f'Mechanism size (mean {nsp.mean():.0f}/160 = {100*(1-nsp.mean()/160):.0f}% reduction)')

plt.tight_layout()
plt.savefig('results/ch3ocho_final/CH3OCHO_summary.png', dpi=120, bbox_inches='tight')
print('saved results/ch3ocho_final/CH3OCHO_summary.png')
