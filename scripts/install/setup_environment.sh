#!/bin/bash

# Setup script for MINT environment
set -e

echo "Setting up MINT environment..."

# Create conda environment in local folder
echo "Creating conda environment in .env folder..."
conda env create --file=scripts/install/environment.yml --prefix=./.env

# Activate the environment and install the package in editable mode
echo "Installing MINT package in editable mode..."
conda run --prefix=./.env pip install -e .

# Test the installation
echo "Testing installation..."
conda run --prefix=./.env python -c "import mint; print('Success')"

echo "Environment setup complete!"
echo "To activate the environment, use: conda activate ./.env"
echo "To run commands in the environment, use: conda run --prefix=./.env <command>" 