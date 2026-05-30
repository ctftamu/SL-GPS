"""
SL-GPS Validation Pipeline — AramcoMech 3.0
=============================================
Orchestrates the full validation workflow:
  1. Generate training data (GPS simulations)
  2. Train neural network
  3. Run validation cases (detailed + SL-GPS)
  4. Generate comparison plots
  5. Compile LaTeX report

Usage:
    python run_all.py          # Run full pipeline
    python run_all.py --from 3 # Resume from step 3
"""

import sys
import os
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("01_generate_data.py", "Training Data Generation (GPS simulations)"),
    ("02_train_model.py",   "Neural Network Training"),
    ("03_validate.py",      "Validation Simulations (detailed + SL-GPS)"),
    ("04_plot_results.py",  "Plot Generation"),
]


def run_step(script, description, step_num):
    """Run a pipeline step and report timing."""
    print(f"\n{'═' * 70}")
    print(f"  STEP {step_num}: {description}")
    print(f"  Script: {script}")
    print(f"{'═' * 70}\n")

    script_path = os.path.join(BASE_DIR, script)
    if not os.path.isfile(script_path):
        print(f"  ERROR: Script not found: {script_path}")
        return False

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ✗ STEP {step_num} FAILED (exit code {result.returncode})")
        print(f"    Elapsed: {elapsed:.1f}s")
        return False
    else:
        print(f"\n  ✓ STEP {step_num} COMPLETE")
        print(f"    Elapsed: {elapsed:.1f}s")
        return True


def compile_latex():
    """Attempt to compile the LaTeX report."""
    print(f"\n{'═' * 70}")
    print(f"  STEP 5: LaTeX Report Compilation")
    print(f"{'═' * 70}\n")

    docs_dir = BASE_DIR
    tex_file = os.path.join(docs_dir, 'report.tex')

    if not os.path.isfile(tex_file):
        print("  WARNING: report.tex not found, skipping compilation")
        return

    # Check if pdflatex is available
    try:
        subprocess.run(['pdflatex', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  WARNING: pdflatex not found. Skipping LaTeX compilation.")
        print(f"  To compile manually: cd {docs_dir} && pdflatex report.tex")
        return

    # Run pdflatex twice for references/ToC
    for i in range(2):
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', 'report.tex'],
            cwd=docs_dir,
            capture_output=True
        )

    if os.path.isfile(os.path.join(docs_dir, 'report.pdf')):
        print(f"  ✓ Report compiled: {os.path.join(docs_dir, 'report.pdf')}")
    else:
        print("  WARNING: PDF not generated. Check LaTeX errors.")
        print(f"  Compile manually: cd {docs_dir} && pdflatex report.tex")


if __name__ == '__main__':
    # Parse --from argument
    start_step = 1
    if '--from' in sys.argv:
        idx = sys.argv.index('--from')
        if idx + 1 < len(sys.argv):
            start_step = int(sys.argv[idx + 1])

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     SL-GPS VALIDATION PIPELINE — AramcoMech 3.0                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Working directory: {BASE_DIR}")
    print(f"  Starting from step: {start_step}")
    print(f"  Total steps: {len(STEPS)} + LaTeX compilation")

    # Check mechanism file
    mech_file = os.path.join(BASE_DIR, 'aramco_v3.cti')
    if not os.path.isfile(mech_file):
        print(f"\n  ⚠ WARNING: Mechanism file not found: aramco_v3.cti")
        print("    Download from: https://www.universityofgalway.ie/combustionchemistrycentre/mechanismdownloads/")
        print("    Convert CHEMKIN → CTI: ck2cti --input=chem.inp --thermo=therm.dat --output=aramco_v3.cti")
        if start_step <= 1:
            print("\n    Cannot proceed without mechanism file.")
            sys.exit(1)

    overall_start = time.time()
    failed = False

    for i, (script, description) in enumerate(STEPS, 1):
        if i < start_step:
            print(f"\n  Skipping step {i}: {description}")
            continue

        success = run_step(script, description, i)
        if not success:
            failed = True
            print(f"\n  Pipeline halted at step {i}.")
            print(f"  Fix the issue and resume with: python run_all.py --from {i}")
            break

    if not failed:
        compile_latex()

    overall_elapsed = time.time() - overall_start
    print(f"\n{'═' * 70}")
    if failed:
        print(f"  PIPELINE INCOMPLETE — failed at step {i}")
    else:
        print(f"  ✓ PIPELINE COMPLETE")
    print(f"  Total time: {overall_elapsed/60:.1f} minutes")
    print(f"{'═' * 70}")
