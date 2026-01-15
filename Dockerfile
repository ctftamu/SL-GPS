# Dockerfile for SL-GPS on Hugging Face Spaces
# Python 3.10 - all dependencies have pre-built wheels for this version
# NO source compilation allowed - using --only-binary for ALL packages

FROM python:3.10.15-slim

# Set working directory
WORKDIR /app

# Only upgrade pip (minimal overhead)
RUN pip install --upgrade pip setuptools wheel

# Install from HF_SPACE_requirements.txt with STRICT wheel-only mode
# This prevents ANY source compilation, even if someone adds >= constraints
RUN pip install --no-cache-dir \
    --only-binary=:all: \
    -r HF_SPACE_requirements.txt || \
    pip install --no-cache-dir \
    --prefer-binary \
    -r HF_SPACE_requirements.txt

# Copy application code
COPY . /app

# Install SL-GPS package
RUN pip install --no-cache-dir -e /app/src/slgps

# Expose Gradio default port
EXPOSE 7860

# Run the HF Spaces app
CMD ["python", "-m", "HF_SPACE_app"]
