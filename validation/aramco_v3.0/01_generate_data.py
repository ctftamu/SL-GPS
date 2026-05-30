"""
Step 1: Generate Training Data for AramcoMech 3.0
==================================================
Runs parallel autoignition simulations with GPS over a wide range of
temperature, pressure, and composition conditions.

Output: data/train/data.csv, data/train/species.csv, threshold files
"""

import sys
import os

# Add src to path for slgps imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from slgps.make_data_parallel import make_data_parallel

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fuel species
fuel = 'CH4'

# Mechanism file (must be in this directory or provide full path)
mech_file = os.path.join(os.path.dirname(__file__), 'mechanism', 'aramco_v3.cti')

# Simulation termination: HRR threshold below which sim ends after ignition
end_threshold = 2e5

# Division of max HRR to define ignition interval
ign_HRR_threshold_div = 300

# GPS resolution: number of intervals during ignition / normal phases
ign_GPS_resolution = 200
norm_GPS_resolution = 40

# Number of GPS evaluations per interval
GPS_per_interval = 4

# Number of training simulations
n_cases = 200

# Training ranges
t_rng = [800, 2200]       # Temperature range (K)
p_rng = [0.0, 1.6]        # Pressure range: log10(atm), i.e. 1-40 atm

# Equivalence ratio range (placeholder; species_ranges used instead)
phi_rng = [0.5, 2.0]

# GPS alpha threshold (pathway importance)
alpha = 0.001

# Species composition ranges for randomized initial conditions
species_ranges = {
    'CH4': (0.02, 0.15),
    'O2':  (0.05, 0.30),
    'N2':  (0.50, 0.80),
    'CO2': (0.00, 0.01),
    'H2O': (0.00, 0.05),
}

# Thresholds for always/never species classification
always_threshold = 0.99
never_threshold = 0.01

# Output path
data_path = os.path.join(os.path.dirname(__file__), 'data', 'train')

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Verify mechanism file exists
    if not os.path.isfile(mech_file):
        print(f"ERROR: Mechanism file not found at: {mech_file}")
        print("Please download AramcoMech 3.0 from:")
        print("  https://www.universityofgalway.ie/combustionchemistrycentre/mechanismdownloads/")
        print("Convert to CTI format and place as 'aramco_v3.cti' in the mechanism/ directory.")
        sys.exit(1)

    os.makedirs(data_path, exist_ok=True)

    print("=" * 70)
    print("SL-GPS Training Data Generation — AramcoMech 3.0")
    print("=" * 70)
    print(f"  Fuel:          {fuel}")
    print(f"  Mechanism:     {mech_file}")
    print(f"  Simulations:   {n_cases}")
    print(f"  T range:       {t_rng[0]}–{t_rng[1]} K")
    print(f"  P range:       {10**p_rng[0]:.1f}–{10**p_rng[1]:.1f} atm")
    print(f"  Alpha:         {alpha}")
    print(f"  Output:        {data_path}")
    print("=" * 70)

    make_data_parallel(
        fuel=fuel,
        mech_file=mech_file,
        end_threshold=end_threshold,
        ign_HRR_threshold_div=ign_HRR_threshold_div,
        ign_GPS_resolution=ign_GPS_resolution,
        norm_GPS_resolution=norm_GPS_resolution,
        GPS_per_interval=GPS_per_interval,
        n_cases=n_cases,
        t_rng=t_rng,
        p_rng=p_rng,
        phi_rng=phi_rng,
        alpha=alpha,
        always_threshold=always_threshold,
        never_threshold=never_threshold,
        pathname=data_path,
        species_ranges=species_ranges
    )

    print("\n✓ Training data generation complete.")
    print(f"  Data saved to: {data_path}")
