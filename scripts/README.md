# Scripts Directory

This directory contains utility scripts for the MINT project.

## Installation Scripts

### `install/` Directory
Contains all installation-related files and scripts:

- **`setup_environment.sh`** - Automated environment setup script
- **`activate_mint.sh`** - Environment activation helper
- **`environment.yml`** - Conda environment specification

**Usage:**
```bash
./scripts/install/setup_environment.sh
```

## Data Processing Scripts

### `counting_pseudomultimers.py`
Script for counting pseudomultimers in protein data.

### `listing_pseudomultimers.py`
Script for listing pseudomultimers in protein data.

### `data-processing-environment.yml`
Environment file for data processing tasks.

## Environment Management

The scripts use a local conda environment located at `./.env` to avoid conflicts with system packages and other projects. 