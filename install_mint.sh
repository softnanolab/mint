#!/bin/bash

conda env create --name mint --file=environment.yml

# Install mmseqs2 inside the 'mint' environment
conda run -n mint conda install -y -c bioconda -c conda-forge mmseqs2

# Ensure pip operations run inside the 'mint' environment
conda run -n mint pip uninstall -y torch
conda run -n mint pip install torch
conda run -n mint pip install fire
conda run -n mint pip install wandb
conda run -n mint pip install seaborn
conda run -n mint pip install -e .