# Setup & Installation

SL-GPS is a Python 3 framework for chemistry reduction using neural networks and GPS. This guide covers installation and verification.

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB (8GB+ recommended for parallel training)
- **Disk**: ~1GB for dependencies and data

## Installation Methods

### Method 1: Quick Install (Recommended)

Clone and install from source with all dependencies:

```bash
# Clone the repository
git clone https://github.com/ctftamu/SL-GPS.git
cd SL-GPS

# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Method 2: Step-by-Step Installation

If you prefer to install packages individually:

```bash
# Upgrade pip
pip install --upgrade pip

# 1. Install base scientific libraries
pip install numpy==1.26.4 matplotlib pandas

# 2. Install machine learning libraries
pip install tensorflow scikit-learn

# 3. Install Cantera (CRITICAL: exact version required)
pip install --no-cache-dir "cantera==2.6.0"

# 4. Install other dependencies
pip install networkx joblib

# 5. Install SL-GPS
pip install git+https://github.com/ctftamu/SL-GPS.git
```

## Virtual Environment Setup (Recommended)

Using a virtual environment isolates this project's dependencies:

### Using `venv` (Python 3.8+)

```bash
# Create virtual environment
python3 -m venv sl_gps_env

# Activate it
source sl_gps_env/bin/activate          # Linux/macOS
# OR
sl_gps_env\Scripts\activate             # Windows PowerShell

# Then install dependencies
pip install -r requirements.txt
```

### Using `conda` (if you have Anaconda/Miniconda)

```bash
# Create conda environment
conda create -n sl_gps python=3.10

# Activate it
conda activate sl_gps

# Install dependencies
pip install -r requirements.txt
```

## Dependency Versions

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| **Python** | 3.8+ | Language | - |
| **cantera** | 2.6.0 | Chemical kinetics | **EXACT VERSION** - 2.6.1+ breaks compatibility |
| **tensorflow** | 2.x | Neural networks | 2.13+ recommended for Python 3.10+ |
| **numpy** | 1.26.4 | Numerics | Pinned for stability |
| **scikit-learn** | latest | Data scaling | For MinMaxScaler |
| **networkx** | latest | Graph algorithms | For GPS flux graphs |
| **pandas** | latest | Data I/O | For CSV handling |
| **joblib** | latest | Parallelization | For ensemble training |
| **mkdocs** | latest | Documentation | For serving docs locally |
| **mkdocs-material** | latest | Docs theme | - |

## Verification

After installation, verify everything works:

### Quick Test

```python
python -c "import cantera; import tensorflow; import numpy; print('All imports successful!')"
```

### Run Unit Tests

```bash
# Navigate to repo directory
cd SL-GPS

# Run tests
python -m pytest tests/test_basic.py -v
```

### Test Data Generation (Optional)

For a quick end-to-end test, edit `src/slgps/main.py`:

```python
# Change n_cases to a small number for testing
n_cases = 5  # Instead of 100

# Run
python src/slgps/main.py
```

This will generate a small training dataset and train an ANN (~5-10 minutes).

## Troubleshooting

### Issue: Cantera Installation Fails

**Problem**: `pip install cantera` fails or installs wrong version

**Solution**:
```bash
# Force exact version
pip install --no-cache-dir "cantera==2.6.0"

# Verify installation
python -c "import cantera; print(cantera.__version__)"
# Should print: 2.6.0
```

### Issue: TensorFlow Won't Install

**Problem**: `pip install tensorflow` fails with CUDA/GPU errors

**Solution** (CPU-only, faster to install):
```bash
pip install tensorflow-cpu
```

**Solution** (with GPU support, requires CUDA):
```bash
# First install CUDA toolkit appropriate for your GPU
# Then:
pip install tensorflow[and-cuda]
```

### Issue: NumPy Version Conflicts

**Problem**: `ImportError: numpy version mismatch`

**Solution**:
```bash
pip install --upgrade --force-reinstall numpy==1.26.4
```

### Issue: ModuleNotFoundError in GPS submodule

**Problem**: `from slgps.GPS.src.* import *` fails

**Solution**:
```bash
# Make sure you're in the repo directory
cd /path/to/SL-GPS

# Reinstall from source
pip install -e .
```

### Issue: GPU/CUDA-related Warnings

**Solution** (safe to ignore):
```python
# Add to top of your Python scripts to suppress TensorFlow GPU warnings:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import tensorflow as tf
```

## Windows-Specific Notes

### Cantera Installation on Windows

```bash
# Use pre-built wheels (easier)
pip install --no-cache-dir cantera==2.6.0
```

### Virtual Environment Activation

```cmd
# Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1
```

## Next Steps

After successful installation:

1. **Read the [Usage Workflow](workflow.md)** - Step-by-step guide for data generation and training
2. **Check [Code Structure](code_structure.md)** - How to customize ANN architecture and GPS parameters
3. **Try the examples** - Run the provided test cases in `tests/`

## Common Questions

**Q: Can I use Cantera 2.7 or newer?**
A: No. Cantera 2.6.0 is required. API changes in newer versions break compatibility. Use exactly 2.6.0.

**Q: Do I need a GPU?**
A: No, but it's faster. TensorFlow runs on CPU by default. For GPU support, install TensorFlow with CUDA.

**Q: What Python version should I use?**
A: 3.8-3.11 are tested. 3.10 is recommended for best compatibility with all dependencies.

**Q: Can I use conda?**
A: Yes, but Cantera installation is easier with pip. Mix both if needed: `conda create && pip install`.

---

**Still having issues?** Check the [GitHub Issues](https://github.com/ctftamu/SL-GPS/issues) or contact us via [Discord](https://discord.com/channels/1333609076726431798/1333610748424880128).
