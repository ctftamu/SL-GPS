"""
Step 4: Generate Validation Plots — AramcoMech 3.0
====================================================
Creates comparison plots between detailed and SL-GPS simulations for each
validation case. Generates:
  - Temperature profiles (detailed vs SL-GPS)
  - Heat release rate profiles
  - Species evolution for key species
  - Mechanism size (species/reactions count over time)
  - Summary: ignition delay comparison across all cases

Output: plots/*.pdf
"""

import sys
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir = os.path.join(os.path.dirname(__file__), 'results', 'ch4')
plots_dir = os.path.join(os.path.dirname(__file__), 'results', 'ch4')

# Key species to plot
plot_species = ['CH4', 'O2', 'CO2', 'H2O', 'CO', 'OH']

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def find_ignition_delay(time_array, temp_array):
    """Find ignition delay as time of maximum dT/dt."""
    if len(time_array) < 3:
        return None
    dt = np.diff(time_array)
    dT = np.diff(temp_array)
    dTdt = dT / dt
    idx = np.argmax(dTdt)
    return time_array[idx]


def plot_case_comparison(case_data, save_path):
    """Generate a multi-panel comparison plot for one validation case."""
    config = case_data['config']
    det = case_data.get('detailed')
    sl = case_data.get('slgps')

    if det is None and sl is None:
        print(f"  Skipping {config['label']}: no data")
        return

    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(
        f"SL-GPS Validation: {config['label']}\n"
        f"T₀={config['T0']} K, P={config['P_atm']} atm, φ={config['phi']}",
        fontsize=13, fontweight='bold'
    )

    # --- Panel 1: Temperature ---
    ax = axs[0]
    if det is not None:
        det_dict = det[0]
        ax.plot(det_dict['axis0'], det_dict['temperature'], 'k-', lw=1.5, label='Detailed')
    if sl is not None:
        sl_dict = sl[0]
        ax.plot(sl_dict['axis0'], sl_dict['temperature'], 'r--', lw=1.5, label='SL-GPS')
    ax.set_ylabel('Temperature (K)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Heat Release Rate ---
    ax = axs[1]
    if det is not None:
        ax.plot(det_dict['axis0'], det_dict['heat_release_rate'], 'k-', lw=1.5, label='Detailed')
    if sl is not None:
        ax.plot(sl_dict['axis0'], sl_dict['heat_release_rate'], 'r--', lw=1.5, label='SL-GPS')
    ax.set_ylabel('HRR (J/m³·s)')
    ax.set_yscale('symlog', linthresh=1e3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Key Species ---
    ax = axs[2]
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_species)))
    for j, spec in enumerate(plot_species):
        if det is not None and spec in det_dict.get('species_names', []):
            idx = det_dict['species_names'].index(spec) if 'species_names' in det_dict else j
            # For mole_fraction stored as 2D array
            if hasattr(det_dict.get('mole_fraction'), 'shape') and len(det_dict['mole_fraction'].shape) == 2:
                ax.plot(det_dict['axis0'], det_dict['mole_fraction'][:, idx],
                        '-', color=colors[j], lw=1.2, label=f'{spec} (det)')
        if sl is not None and hasattr(sl_dict.get('mole_fraction'), 'shape'):
            if 'species_names' in sl_dict and spec in sl_dict['species_names']:
                idx_sl = sl_dict['species_names'].index(spec)
                ax.plot(sl_dict['axis0'], sl_dict['mole_fraction'][:, idx_sl],
                        '--', color=colors[j], lw=1.2, label=f'{spec} (SL)')
    ax.set_ylabel('Mole Fraction')
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Mechanism Size ---
    ax = axs[3]
    if sl is not None:
        mech_times = sl[1]
        rxn_nums = sl[2]
        spec_nums = sl[3]
        ax.plot(mech_times, spec_nums, 'g-o', ms=3, lw=1, label='# Species')
        ax2 = ax.twinx()
        ax2.plot(mech_times, rxn_nums, 'b-s', ms=3, lw=1, label='# Reactions')
        ax2.set_ylabel('# Reactions', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
    ax.set_ylabel('# Species', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_ignition_delay_summary(all_cases, save_path):
    """Bar chart comparing ignition delays across all validation cases."""
    labels = []
    ign_det = []
    ign_sl = []

    for case_data in all_cases:
        config = case_data['config']
        det = case_data.get('detailed')
        sl = case_data.get('slgps')

        labels.append(config['label'])

        if det is not None:
            det_dict = det[0]
            tau_det = find_ignition_delay(
                np.array(det_dict['axis0']),
                np.array(det_dict['temperature'])
            )
            ign_det.append(tau_det if tau_det else 0)
        else:
            ign_det.append(0)

        if sl is not None:
            sl_dict = sl[0]
            tau_sl = find_ignition_delay(
                np.array(sl_dict['axis0']),
                np.array(sl_dict['temperature'])
            )
            ign_sl.append(tau_sl if tau_sl else 0)
        else:
            ign_sl.append(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, np.array(ign_det)*1000, width, label='Detailed', color='black', alpha=0.7)
    bars2 = ax.bar(x + width/2, np.array(ign_sl)*1000, width, label='SL-GPS', color='red', alpha=0.7)

    ax.set_ylabel('Ignition Delay (ms)')
    ax.set_title('Ignition Delay Comparison — AramcoMech 3.0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_reduction_summary(all_cases, save_path, mech_file=None):
    """Plot showing mechanism reduction ratio for each case."""
    labels = []
    avg_species = []
    avg_reactions = []

    # Get full mechanism size
    try:
        import cantera as ct
        if mech_file:
            soln = ct.Solution(mech_file)
            full_n_spec = len(soln.species_names)
            full_n_rxn = len(soln.reactions())
        else:
            full_n_spec = 253  # AramcoMech 3.0 approximate
            full_n_rxn = 1542
    except Exception:
        full_n_spec = 253
        full_n_rxn = 1542

    for case_data in all_cases:
        config = case_data['config']
        sl = case_data.get('slgps')
        labels.append(config['label'])

        if sl is not None and len(sl) > 3:
            spec_nums = sl[3]
            rxn_nums = sl[2]
            avg_species.append(np.mean(spec_nums))
            avg_reactions.append(np.mean(rxn_nums))
        else:
            avg_species.append(0)
            avg_reactions.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(labels))

    # Species reduction
    ax1.bar(x, avg_species, color='green', alpha=0.7)
    ax1.axhline(full_n_spec, color='k', ls='--', lw=1.5, label=f'Full mechanism ({full_n_spec})')
    ax1.set_ylabel('Avg # Species in Reduced Mechanism')
    ax1.set_title('Species Reduction')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Reaction reduction
    ax2.bar(x, avg_reactions, color='blue', alpha=0.7)
    ax2.axhline(full_n_rxn, color='k', ls='--', lw=1.5, label=f'Full mechanism ({full_n_rxn})')
    ax2.set_ylabel('Avg # Reactions in Reduced Mechanism')
    ax2.set_title('Reaction Reduction')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    os.makedirs(plots_dir, exist_ok=True)

    # Load all result files
    result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.pkl')])

    if not result_files:
        print(f"ERROR: No result files found in {results_dir}")
        print("Run 03_validate.py first.")
        sys.exit(1)

    print("=" * 70)
    print("SL-GPS Plot Generation — AramcoMech 3.0")
    print("=" * 70)
    print(f"  Found {len(result_files)} result files")
    print("=" * 70)

    all_cases = []
    for fname in result_files:
        fpath = os.path.join(results_dir, fname)
        with open(fpath, 'rb') as f:
            case_data = pickle.load(f)
        all_cases.append(case_data)

        # Individual case plot
        label = case_data['config']['label']
        plot_path = os.path.join(plots_dir, f'{label}.png')
        print(f"\n  Plotting: {label}")
        plot_case_comparison(case_data, plot_path)

    # Summary plots
    print(f"\n  Generating summary plots...")
    plot_ignition_delay_summary(
        all_cases,
        os.path.join(plots_dir, 'ignition_delay_summary.png')
    )

    mech_file_path = os.path.join(os.path.dirname(__file__), 'mechanism', 'aramco_v3.cti')
    plot_reduction_summary(
        all_cases,
        os.path.join(plots_dir, 'reduction_summary.png'),
        mech_file=mech_file_path if os.path.isfile(mech_file_path) else None
    )

    print("\n" + "=" * 70)
    print("✓ All plots generated.")
    print(f"  Plots saved to: {plots_dir}")
    print("=" * 70)
