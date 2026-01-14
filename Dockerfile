# Dockerfile for SL-GPS on Hugging Face Spaces
# Optimized with layer caching to avoid rebuilding heavy dependencies

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Update pip and install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# LAYER 1: Install heavy dependencies first (TensorFlow, Cantera)
# These take the longest and should be cached separately from app code
RUN pip install --no-cache-dir \
    tensorflow>=2.13.0 \
    cantera==2.6.0

# LAYER 2: Install other core dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scikit-learn>=1.3.0 \
    networkx>=3.0 \
    pandas>=2.0.0 \
    joblib>=1.3.0 \
    matplotlib>=3.7.0 \
    gradio==4.29.0

# LAYER 3: Copy application code (changes most frequently)
COPY . /app

# LAYER 4: Install SL-GPS package and any additional requirements
RUN pip install -e /app/src/slgps

# Expose Gradio default port
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860 || exit 1

# Run the HF Spaces app
CMD ["python", "-m", "HF_SPACE_app"]
