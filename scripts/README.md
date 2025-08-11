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
**Independent script** for counting pseudomultimers in protein data. This script contains the complete `ProcessingCATH` class and can be run independently.

**Features:**
- Counts chains with contiguous domains
- Counts domains with different numbers of segments
- Plots frequency distributions of pseudomultimers
- Calculates pseudomultimer statistics

**Usage:**
```bash
conda run --prefix=./.env python scripts/counting_pseudomultimers.py --help
conda run --prefix=./.env python scripts/counting_pseudomultimers.py --database_path=path/to/data.txt
```

### `listing_pseudomultimers.py`
**Independent script** for listing pseudomultimers in protein data. This script contains both the `ProcessingCATH` class and the `ListingPseudomultimers` class and can be run independently.

**Features:**
- Creates detailed dictionaries of chain-level pseudomultimers
- Generates JSON files with pseudomultimer information
- Plots histograms of domain lengths
- Inherits all functionality from `ProcessingCATH`

**Usage:**
```bash
conda run --prefix=./.env python scripts/listing_pseudomultimers.py --help
conda run --prefix=./.env python scripts/listing_pseudomultimers.py --data=path/to/data.txt
```

### `data-processing-environment.yml`
Environment file for data processing tasks.

## Script Independence

Both `counting_pseudomultimers.py` and `listing_pseudomultimers.py` are now **completely independent**:

- ✅ **No cross-dependencies** - Each script contains all necessary classes and functions
- ✅ **Self-contained** - Can be run individually without importing from other scripts
- ✅ **Same functionality** - Both scripts provide the same core functionality as before
- ✅ **Easy maintenance** - Changes to one script don't affect the other

## Environment Management

The scripts use a local conda environment located at `./.env` to avoid conflicts with system packages and other projects. 