# Dockerfile for SL-GPS on Hugging Face Spaces
# Python 3.10 - all dependencies have pre-built wheels for this version
# NO source compilation needed

FROM python:3.10.15-slim

# Set working directory
WORKDIR /app

# Only upgrade pip (minimal overhead)
RUN pip install --upgrade pip setuptools wheel

# LAYER 1: Install TensorFlow first (uses pre-built wheels, no compilation)
# Only installs CPU version to reduce size and build time
RUN pip install --no-cache-dir \
    tensorflow-cpu==2.13.0

# LAYER 2: Install Cantera from conda-forge wheels (faster than compiling)
# Install from PyPI pre-built wheel
RUN pip install --no-cache-dir \
    cantera==2.6.0 \
    --only-binary cantera || pip install --no-cache-dir cantera==2.6.0

# LAYER 3: Install other dependencies (all pre-built wheels)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    networkx==3.3 \
    pandas==2.2.3 \
    joblib==1.4.2 \
    matplotlib==3.9.2 \
    gradio==4.29.0

# LAYER 4: Copy application code (changes most frequently)
COPY . /app

# LAYER 5: Install SL-GPS package
RUN pip install --no-cache-dir -e /app/src/slgps

# Expose Gradio default port
EXPOSE 7860

# Run the HF Spaces app
CMD ["python", "-m", "HF_SPACE_app"]
