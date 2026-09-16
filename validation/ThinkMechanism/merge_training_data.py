"""
Merge base (train_ch3ocho) + expanded (train_ch3ocho_expanded) training data and
recompute always/never/var partitions over the union.

Rationale: expanded NTC-stratified data starved high-T samples (7.9% at T>=1300K),
pushing GPS-required high-T species (CH, C, ...) into the 'never' list. Merging
restores high-T coverage while keeping NTC coverage.

Output: data/train_ch3ocho_merged/{data.csv,species.csv,*_spec_nums.csv}
(same formats as make_data_parallel).
"""
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
# usage: merge_training_data.py [out_name] [set1 set2 ...] (names under data/)
OUT_NAME = sys.argv[1] if len(sys.argv) > 1 else 'train_ch3ocho_merged'
SET_NAMES = sys.argv[2:] if len(sys.argv) > 2 else ['train_ch3ocho', 'train_ch3ocho_expanded']
SETS = [os.path.join(BASE, 'data', n) for n in SET_NAMES]
OUT = os.path.join(BASE, 'data', OUT_NAME)
ALWAYS_THRESHOLD = 0.99
# never = only species GPS never selected in ANY sample; a frequency cutoff here
# silently vetoes rare-but-required species when the case mix is unbalanced
NEVER_THRESHOLD = 0.0

os.makedirs(OUT, exist_ok=True)


def load_names(path, fname):
    names = pd.read_csv(os.path.join(path, fname)).columns.to_list()[:-1]
    if names and names[0][:2] == '# ':
        names[0] = names[0][2:]
    return names


datas, masks = [], []
all_species = None
for d in SETS:
    data = pd.read_csv(os.path.join(d, 'data.csv'))
    var = load_names(d, 'var_spec_nums.csv')
    always = load_names(d, 'always_spec_nums.csv')
    sp = pd.read_csv(os.path.join(d, 'species.csv')).iloc[:, :-1]
    sp.columns = var if len(sp.columns) == len(var) else sp.columns
    # fix '# ' prefix on first species.csv column
    sp.columns = [c[2:] if c[:2] == '# ' else c for c in sp.columns]

    # full mechanism species order = data.csv columns after T, atm
    full_species = [c for c in data.columns[2:]]
    if all_species is None:
        all_species = full_species
    assert full_species == all_species, f'species order mismatch in {d}'

    # reconstruct full binary mask: var from file, always=1, never(rest)=0
    full_mask = pd.DataFrame(0, index=range(len(data)), columns=full_species, dtype=int)
    for s in always:
        if s in full_mask.columns:
            full_mask[s] = 1
    for s in sp.columns:
        if s in full_mask.columns:
            full_mask[s] = sp[s].astype(int)
    datas.append(data)
    masks.append(full_mask)
    print(f"{os.path.basename(d)}: rows={len(data)} var={len(var)} always={len(always)}")

data_m = pd.concat(datas, ignore_index=True)
mask_m = pd.concat(masks, ignore_index=True)
freq = mask_m.mean()

always_new = [s for s in all_species if freq[s] > ALWAYS_THRESHOLD]
never_new = [s for s in all_species if freq[s] <= NEVER_THRESHOLD]
var_new = [s for s in all_species if NEVER_THRESHOLD < freq[s] <= ALWAYS_THRESHOLD]
print(f"\nmerged: rows={len(data_m)}  var={len(var_new)} always={len(always_new)} never={len(never_new)}")
for s in ['CH', 'C', 'CH3', 'CH3OCO', 'CH2OCHO']:
    part = 'always' if s in always_new else ('never' if s in never_new else 'var')
    print(f"  {s:9s} freq={freq[s]*100:6.2f}% -> {part}")

# --- write outputs in make_data_parallel formats -------------------------------
data_header = ','.join(['Temperature', 'Atmospheres'] + all_species)
np.savetxt(os.path.join(OUT, 'data.csv'), data_m.to_numpy(), delimiter=',', header=data_header)

var_header = ''.join(s + ',' for s in var_new)[:-1] + ','  # trailing comma -> extra col like original
np.savetxt(os.path.join(OUT, 'species.csv'), mask_m[var_new].to_numpy(), delimiter=',', header=var_header)

for fname, names in [('var_spec_nums.csv', var_new),
                     ('always_spec_nums.csv', always_new),
                     ('never_spec_nums.csv', never_new)]:
    with open(os.path.join(OUT, fname), 'w') as f:
        f.write(''.join(s + ',' for s in names) + '\n')

print(f"\nwritten to {OUT}")
