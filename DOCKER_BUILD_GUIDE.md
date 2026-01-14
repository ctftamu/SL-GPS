# Docker Build Optimization Guide

## Overview

The Dockerfile is optimized to handle build timeouts on Hugging Face Spaces by:

1. **Layer Caching**: Heavy dependencies (TensorFlow, Cantera) are installed in separate layers
2. **Build Context Optimization**: `.dockerignore` excludes unnecessary files, speeding up build context transfer
3. **Minimal Base Image**: Uses `python:3.10-slim` instead of full Python image

## How It Works

### Layer Strategy

```
Layer 1: Update apt & install build tools (rarely changes)
Layer 2: Install TensorFlow + Cantera (takes ~10-15 min, cached after first build)
Layer 3: Install other dependencies (takes ~2 min, cached after first build)
Layer 4: Copy application code (changes frequently, quick layer)
Layer 5: Install SL-GPS package (fast, depends on app code)
```

**Key insight**: If you rebuild the image after code changes, Docker reuses Layers 1-3 from cache. Only Layers 4-5 are rebuilt, which is much faster.

## Building Locally

```bash
# Build the image
docker build -t sl-gps:latest .

# Run the container
docker run -p 7860:7860 sl-gps:latest

# The app will be accessible at http://localhost:7860
```

## On Hugging Face Spaces

1. Push the Dockerfile to your repo
2. In Spaces settings, configure:
   - **Dockerfile**: Set to use your Dockerfile
   - **Space Hardware**: Use "CPU" or "T4 GPU" (more resources = faster builds)
   - **Build timeout**: HF Spaces typically allows 30-60 minutes

3. Spaces will automatically detect the Dockerfile and use it for builds

## Build Time Reduction

**Before optimization:**
- Full rebuild: ~25-30 minutes (rebuilding TensorFlow + Cantera every time)

**After optimization:**
- First build: ~25-30 minutes (same, dependencies must build)
- Subsequent builds: ~3-5 minutes (dependencies cached, only app code rebuilt)

## Troubleshooting

**If build still times out:**
1. Try with "T4 GPU" hardware (Spaces option) instead of CPU
2. Split the `HF_SPACE_requirements.txt` into core + optional dependencies
3. Use pre-built wheels for Cantera (if available for your Python version)

**If you get "No module named HF_SPACE_app":**
- Make sure the Dockerfile's CMD matches your entry point
- Current setup runs: `python -m HF_SPACE_app`

**Monitor build status:**
- Check Spaces "Build logs" tab for real-time build progress
- Look for which layer is taking longest (usually TensorFlow install)

## Notes

- The Dockerfile uses `HF_SPACE_requirements.txt` which already has optimized pinned versions
- Healthcheck is included to verify the app is running
- Port 7860 is standard for Gradio on Hugging Face Spaces
