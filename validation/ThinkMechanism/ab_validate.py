"""
Controlled A/B validation driver for SL-GPS.

Generates training data + trains the ANN once, then runs the validation step
TWICE on the SAME model+data:
  - FIX ON  (SLGPS_FROZEN_STATE_FIX=1): ANN sees the evolving thermochemical state
  - FIX OFF (SLGPS_FROZEN_STATE_FIX=0): original frozen-initial-state behavior

This isolates the effect of the frozen-state runtime fix (which only touches the
SL-GPS runtime, auto_ign_build_SL) with zero retraining.

Usage:
    python ab_validate.py [pipeline_module]
Default pipeline_module = 'run_pipeline' (GRI / Aramco-CH4). For the Aramco
per-fuel pipelines pass e.g. 'run_pipeline_c2h4' or 'run_pipeline_c3h8'.
"""
import os
import sys
import glob
import pickle
import importlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Optional cap on data-gen worker processes (memory-constrained shared node).
_mp = os.environ.get('SLGPS_MAX_PROCS')
if _mp:
    import multiprocessing as _mpmod
    _n = int(_mp)
    _mpmod.cpu_count = lambda: _n
    print(f"[ab] capping multiprocessing.cpu_count() -> {_n}")

MODNAME = sys.argv[1] if len(sys.argv) > 1 else 'run_pipeline'
rp = importlib.import_module(MODNAME)


def ign_delay(result):
    """Time of max dT/dt; None if it never ignites (< 200 K rise)."""
    if result is None:
        return None, 0.0, 0.0
    raw = result[0]
    t = np.asarray(raw['axis0'], dtype=float)
    T = np.asarray(raw['temperature'], dtype=float)
    if t.size < 3:
        return None, float(T.max() - T[0]) if T.size else 0.0, float(T.max()) if T.size else 0.0
    dT = float(T.max() - T[0])
    if dT < 200.0:
        return None, dT, float(T.max())
    dTdt = np.diff(T) / np.diff(t)
    return float(t[int(np.argmax(dTdt))]), dT, float(T.max())


print("=" * 70)
print(f"  A/B VALIDATION DRIVER — pipeline module: {MODNAME}")
print("=" * 70)

# Step 1 + 2: data + model (idempotent; skip if already present)
rp.step1_generate_data()
rp.step2_train()

summary = {}
for mode, val in [('FIX_ON', '1'), ('FIX_OFF', '0')]:
    os.environ['SLGPS_FROZEN_STATE_FIX'] = val
    for p in glob.glob(os.path.join(rp.RESULTS_DIR, '*.pkl')):
        os.remove(p)
    print("\n" + "#" * 70)
    print(f"#  VALIDATION PASS: {mode}  (SLGPS_FROZEN_STATE_FIX={val})")
    print("#" * 70)
    rp.step3_validate()

    rows = []
    for case in rp.VALIDATION_CASES:
        label = case[0]
        p = os.path.join(rp.RESULTS_DIR, f'{label}.pkl')
        if not os.path.isfile(p):
            rows.append((label, None, None, None, None, None))
            continue
        d = pickle.load(open(p, 'rb'))
        det = d.get('detailed')
        sl = d.get('slgps')
        idd, dTd, Tmd = ign_delay(det)
        ids, dTs, Tms = ign_delay(sl)
        err = (abs(ids - idd) / idd * 100.0) if (idd and ids) else None
        rows.append((label, idd, ids, err, dTd, dTs))
    summary[mode] = rows

# ---- print comparison table --------------------------------------------------
print("\n" + "=" * 78)
print("  IGNITION-DELAY COMPARISON  (err% = |slgps-detailed|/detailed x100)")
print("=" * 78)
hdr = f"{'case':14s} {'det_idt':>11s} | {'FIXOFF idt':>11s} {'err%':>8s} | {'FIXON idt':>11s} {'err%':>8s}"
print(hdr)
print("-" * 78)
off = {r[0]: r for r in summary['FIX_OFF']}
on = {r[0]: r for r in summary['FIX_ON']}
for r in summary['FIX_ON']:
    label = r[0]
    idd = r[1]
    ro = off.get(label)
    rn = on.get(label)
    def fmt_idt(x):
        return f"{x:.4e}" if x else "NO-IGN"
    def fmt_err(x):
        return f"{x:7.1f}%" if x is not None else "   n/a "
    print(f"{label:14s} {fmt_idt(idd):>11s} | {fmt_idt(ro[2]):>11s} {fmt_err(ro[3]):>8s} | "
          f"{fmt_idt(rn[2]):>11s} {fmt_err(rn[3]):>8s}")
print("=" * 78)
print("FIXON = evolving-state ANN input (bug fixed); FIXOFF = original frozen-state")
