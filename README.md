<h1 align="center">
  Learning the language of protein-protein interactions 
</h1>

## 🖥️ 1. Installation 

The easiest way to set up the MINT environment is using our automated setup script:

```bash
# Run the setup script from the project root
./scripts/install/setup_environment.sh
```

This script will:
- Create a conda environment in `./.env` folder
- Install all dependencies including the correct torch version
- Install the MINT package in editable mode
- Test the installation automatically

### Using the Environment

**Activate the environment:**
```bash
conda activate ./.env
```

**Or use the activation script:**
```bash
source ./scripts/install/activate_mint.sh
```

**Run commands without activating:**
```bash
conda run --prefix=./.env <your-command>
```

**Test the installation:**
```bash
conda run --prefix=./.env python -c "import mint; print('Success')"
```

## 2. Setting up the data

For entries in the PDB, domains have been labelled by the Orengo Group et al. and the data is available here to download: https://www.cathdb.info/download

## 3. Download Model Checkpoint

Download the model checkpoint and note the file path where it is stored:

```bash
wget https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt
```

