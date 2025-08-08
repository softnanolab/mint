<h1 align="center">
  Learning the language of protein-protein interactions 
</h1>

## 🖥️ Installation (uv, Python 3.12 preferred)

Use uv with the single pyproject.toml to create and install the environment:

```bash
# 1) Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Create and activate a Python 3.12 environment
uv venv --python 3.12
source .venv/bin/activate 

# 3) Install from pyproject.toml
uv pip install -e .

# 4) Verify
python -c "import mint; print('Success')"
```

## 2. Setting up the data

For entries in the PDB, domains have been labelled by the Orengo Group et al. and the data is available here to download: https://www.cathdb.info/download

## 3. Download Model Checkpoint

Download the model checkpoint and note the file path where it is stored:

```bash
wget https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt
```

