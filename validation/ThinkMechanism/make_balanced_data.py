"""
Build a temperature-balanced training set from the pooled data.

Pools all rows from the given sets, then caps rows per temperature bin so no
regime dominates (the seesaw: NTC-heavy data broke high-T, high-T-heavy data
broke mid-T). Partitions (always/never/var) are computed on the FULL pool with
the freq==0 never rule, so rebalancing never blocks species.

Usage: python3 make_balanced_data.py [out_name] [cap_per_bin] [set1 set2 ...]
Defaults: train_ch3ocho_balanced 500 train_ch3ocho train_ch3ocho_expanded train_ch3ocho_hiT
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_NAME = sys.argv[1] if len(sys.argv) > 1 else 'train_ch3ocho_balanced'
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 500
SET_NAMES = sys.argv[3:] if len(sys.argv) > 3 else \
    ['train_ch3ocho', 'train_ch3ocho_expanded', 'train_ch3ocho_hiT']
OUT = os.path.join(BASE, 'data', OUT_NAME)
T_BINS = [0, 800, 950, 1100, 1250, 1400, 3000]
ALWAYS_THRESHOLD = 0.99
RNG = np.random.default_rng(42)

os.makedirs(OUT, exist_ok=True)


def load_names(path, fname):
    names = pd.read_csv(os.path.join(path, fname)).columns.to_list()[:-1]
    if names and names[0][:2] == '# ':
        names[0] = names[0][2:]
    return names


datas, masks = [], []
source_always = set()
all_species = None
for name in SET_NAMES:
    d = os.path.join(BASE, 'data', name)
    data = pd.read_csv(os.path.join(d, 'data.csv'))
    var = load_names(d, 'var_spec_nums.csv')
    always = load_names(d, 'always_spec_nums.csv')
    source_always |= set(always)
    sp = pd.read_csv(os.path.join(d, 'species.csv')).iloc[:, :-1]
    sp.columns = [c[2:] if c[:2] == '# ' else c for c in sp.columns]
    full_species = [c for c in data.columns[2:]]
    if all_species is None:
        all_species = full_species
    assert full_species == all_species, f'species order mismatch in {d}'
    full_mask = pd.DataFrame(0, index=range(len(data)), columns=full_species, dtype=int)
    for s in always:
        if s in full_mask.columns:
            full_mask[s] = 1
    for s in sp.columns:
        if s in full_mask.columns:
            full_mask[s] = sp[s].astype(int)
    datas.append(data)
    masks.append(full_mask)
    print(f"{name}: rows={len(data)}")

data_all = pd.concat(datas, ignore_index=True)
mask_all = pd.concat(masks, ignore_index=True)

# partitions from FULL pool (freq==0 never rule)
# always: freq threshold UNION any source set's always list — a rebalance must not
# demote near-universal species (H2 fell to var this way and broke NTC ignition)
freq = mask_all.mean()
always_new = [s for s in all_species if freq[s] > ALWAYS_THRESHOLD or s in source_always]
never_new = [s for s in all_species if freq[s] == 0]
var_new = [s for s in all_species if s not in always_new and s not in never_new]

# balance rows across T bins
T = data_all['# Temperature']
keep = []
print(f"\nT-bin balance (cap={CAP}):")
for lo, hi in zip(T_BINS[:-1], T_BINS[1:]):
    idx = data_all.index[(T >= lo) & (T < hi)].to_numpy()
    take = idx if len(idx) <= CAP else RNG.choice(idx, CAP, replace=False)
    keep.extend(take.tolist())
    print(f"  [{lo:>4},{hi:>4}): pool={len(idx):5d} kept={len(take):4d}")
keep = sorted(keep)
data_b = data_all.loc[keep]
mask_b = mask_all.loc[keep]
print(f"\nbalanced rows={len(data_b)}  var={len(var_new)} always={len(always_new)} never={len(never_new)}")

data_header = ','.join(['Temperature', 'Atmospheres'] + all_species)
np.savetxt(os.path.join(OUT, 'data.csv'), data_b.to_numpy(), delimiter=',', header=data_header)
var_header = ''.join(s + ',' for s in var_new)[:-1] + ','
np.savetxt(os.path.join(OUT, 'species.csv'), mask_b[var_new].to_numpy(), delimiter=',', header=var_header)
for fname, names in [('var_spec_nums.csv', var_new),
                     ('always_spec_nums.csv', always_new),
                     ('never_spec_nums.csv', never_new)]:
    with open(os.path.join(OUT, fname), 'w') as f:
        f.write(''.join(s + ',' for s in names) + '\n')
print(f"written to {OUT}")
