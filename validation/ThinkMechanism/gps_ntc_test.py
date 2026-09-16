"""
Standalone GPS reduction test for the NTC (low-T) case.

Goal: prove whether GPS can capture low-temperature (cool-flame / NTC)
ignition chemistry, isolated from the SL-GPS neural network.

Compares two GPS species-selection strategies on the SAME detailed
autoignition trajectory:
  (A) cold-point-only  -> mimics the buggy GPS_spec (ind_start=1, ind_end=2):
                          flux graph taken only at the near-initial state.
  (B) full-trajectory  -> flux graph sampled at many points across the
                          induction + ignition period (the intended behavior).

For each strategy we take the union of GPS-kept species, build the reduced
mechanism (sub_mech), run autoignition with it, and report whether it ignites.

Usage:
    python gps_ntc_test.py [T0] [P_atm] [alpha] [K] [n_samples]
Defaults: 750 20 0.001 1 40
"""
import sys
import numpy as np
import cantera as ct

from slgps.GPS.src.core.def_GPS import GPS_algo
from slgps.GPS.src.core.def_build_graph import build_flux_graph
from slgps.utils import sub_mech, auto_ign_build_X0

MECH = 'ThInK_1.0_Mech.yaml'
FUEL = 'CH3OCHO'

T0        = float(sys.argv[1]) if len(sys.argv) > 1 else 750.0
P_ATM     = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
ALPHA     = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
K         = int(sys.argv[4])   if len(sys.argv) > 4 else 1
N_SAMPLES = int(sys.argv[5])   if len(sys.argv) > 5 else 40

ELEMENTS = ['C', 'H', 'O']
SOURCES  = [FUEL, 'O2']
TARGETS  = ['CO2', 'H2O']
END_THRESHOLD = 2e3
T_END = 2.0

print(f"=== GPS NTC test  T0={T0}K  P={P_ATM}atm  phi=1.0  alpha={ALPHA}  K={K}  n_samples={N_SAMPLES} ===")

# --- initial stoichiometric mixture (fuel / air) -------------------------------
s0 = ct.Solution(MECH)
s0.set_equivalence_ratio(1.0, FUEL, 'O2:1, N2:3.76')
X0 = s0.mole_fraction_dict()

# --- detailed autoignition -----------------------------------------------------
det_soln = ct.Solution(MECH)
raw, texec, nsteps = auto_ign_build_X0(det_soln, T0, P_ATM, X0,
                                       end_threshold=END_THRESHOLD, end=T_END,
                                       dir_raw='ign')
T_det = np.asarray(raw['temperature'])
t_det = np.asarray(raw['axis0'])
N = len(t_det)
dT_det = float(T_det.max() - T_det[0])
print(f"[detailed]  steps={N}  T0={T_det[0]:.1f}  Tmax={T_det.max():.1f}  dT={dT_det:.1f}  "
      f"t_end={t_det[-1]:.4e}s  ({nsteps} solver steps)")

# --- GPS species union over a set of trajectory indices ------------------------
def gps_union(indices):
    soln = ct.Solution(MECH)
    species = set(SOURCES) | set(TARGETS) | {'N2'}
    for i0 in indices:
        for e in ELEMENTS:
            fg = build_flux_graph(soln, raw, e, path_save='flux_graph_tmp', overwrite=True,
                                  i0=int(i0), i1='eq', constV=False)
            nodes = set(fg.nodes())
            for sc in SOURCES:
                for tg in TARGETS:
                    if sc in nodes and tg in nodes:
                        res = GPS_algo(soln, fg, sc, tg, path_save=None, K=K,
                                       alpha=ALPHA, beta=0.5, normal='max',
                                       iso=None, overwrite=False, raw='unknown',
                                       notes=None, gamma=None)
                        species |= set(res['species'].keys())
    return species

# --- run reduced mechanism at same conditions ----------------------------------
def run_reduced(species):
    red = sub_mech(MECH, species)
    kept = set(red.species_names)
    X0_red = {k: v for k, v in X0.items() if k in kept}
    raw_r, _, nst = auto_ign_build_X0(red, T0, P_ATM, X0_red,
                                      end_threshold=END_THRESHOLD, end=T_END,
                                      dir_raw='ign')
    Tr = np.asarray(raw_r['temperature'])
    return len(red.species_names), float(Tr.max() - Tr[0]), float(Tr.max()), nst

# --- strategy A: cold point only (current buggy behavior) ----------------------
cold_idx = [1]
sp_cold = gps_union(cold_idx)
nA, dTA, TmaxA, nstA = run_reduced(sp_cold)
print(f"\n[A cold-point]   GPS species={len(sp_cold)}  reduced-mech species={nA}")
print(f"                 reduced Tmax={TmaxA:.1f}  dT={dTA:.1f}  "
      f"{'IGNITES' if dTA > 500 else 'NO IGNITION'}")

# --- strategy B: full trajectory sampling (intended behavior) ------------------
full_idx = sorted(set(np.linspace(1, N - 2, N_SAMPLES).astype(int)))
sp_full = gps_union(full_idx)
nB, dTB, TmaxB, nstB = run_reduced(sp_full)
print(f"\n[B full-traj]    GPS species={len(sp_full)}  reduced-mech species={nB}")
print(f"                 reduced Tmax={TmaxB:.1f}  dT={dTB:.1f}  "
      f"{'IGNITES' if dTB > 500 else 'NO IGNITION'}")

# --- species that full-trajectory keeps but cold-point misses ------------------
extra = sorted(sp_full - sp_cold)
print(f"\n[extra species kept by full-traj only]  count={len(extra)}")
print('  ' + ', '.join(extra))

print("\n=== summary ===")
print(f" detailed        dT={dT_det:7.1f}")
print(f" A cold-point    dT={dTA:7.1f}  ({nA} sp)  {'IGNITES' if dTA>500 else 'NO IGN'}")
print(f" B full-traj     dT={dTB:7.1f}  ({nB} sp)  {'IGNITES' if dTB>500 else 'NO IGN'}")
