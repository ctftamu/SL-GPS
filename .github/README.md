# GitHub Pages Configuration for SL-GPS

This directory contains the GitHub Actions workflow that automatically builds and deploys
the documentation to GitHub Pages whenever changes are pushed to the main branch.

## Setup Instructions

To enable GitHub Pages for this repository:

1. **Go to repository settings**: https://github.com/ctftamu/SL-GPS/settings/pages

2. **Configure the following:**
   - **Source**: Deploy from a branch
   - **Branch**: `gh-pages` (will be created automatically by the workflow)
   - **Folder**: `/ (root)`

3. **Allow GitHub Actions to write to pages**:
   - Go to Settings → Actions → General
   - Under "Workflow permissions", ensure "Read and write permissions" is enabled

4. **The workflow will automatically:**
   - Build the docs using MkDocs whenever you push to `main`
   - Create/update the `gh-pages` branch
   - Deploy to `https://ctftamu.github.io/SL-GPS/`

## Files in this Directory

- `deploy-docs.yml` - GitHub Actions workflow that builds and deploys docs
- `copilot-instructions.md` - AI assistant guidelines for this project
