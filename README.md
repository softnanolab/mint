<h1 align="center">
  Learning the language of protein-protein interactions 
</h1>

## 1. 🖥️ Installation (uv, Python 3.12 preferred)

Use uv with the single pyproject.toml to create and install the environment:

```bash
# 1) Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Create and activate a Python 3.12 environment
uv venv --python 3.12
source .venv/bin/activate 

# 3) Install from pyproject.toml
uv pip install -e .

# (Optional) Install the dev dependencies
uv pip install -e .[dev]

# 4) Verify
python -c "import mint; print('Success')"
```

## 2. Download Data

All the data is kept outside the repository. Please select a directory (ideally on SSD for faster read/write speeds) and download the data there. That location will be referenced as BASE_DATA_DIR in the following instructions.

For entries in the PDB, domains have been labelled by the Orengo Group et al. and the data is available here to download: https://www.cathdb.info/download

Please run pixi shell to activate the environment before running the following commands.

### 1.1 Generate a list of PDB ids to download

We use rcsbsearchapi to filter the PDB ids that match the following filters :

- deposition Date <= 2020-05-01
- resolution ≤ 9Å
- sequence length > 20

To generate the list of PDB ids with default parameters, run

```bash
python scripts/generate_pdb_ids.py --base_data_dir BASE_DATA_DIR
```

### 1.2 Download the PDB database (.cif files)

Run the following command to download unzipped .cif files:

```bash
scripts/download_pdbs.sh -o BASE_DATA_DIR -n 16 -c
```

The script will create a BASE_DATA_DIR/cif_zipped directory with the zipped .cif files if it doesn't exist and parallelly download the zipped .cif files.

You can change the number of CPUs used for downloading by changing the -n flag. The -c flag is used to specify .cif file format (-p for .pdb format).

The script will skip files already present so you can rerun the script if it was interrupted or some files were missing without re-downloading existing files.

### 1.3 Process Data

Run the following command to process the downloaded .cif files:

```bash
python scripts/process_data.py --base_data_dir BASE_DATA_DIR --files_per_folder 1000 --num_cpus 16
```

This script will:
- Unzip and organize files into folders of 1000 files each for faster read/write speeds
- Analyze each structure and extract features
- Create a BASE_DATA_DIR/pdb_features.json file with the dataset features
- Create a BASE_DATA_DIR/failed_files.json file with any files that failed processing

The script keeps the original zipped files so you can rerun if needed without re-downloading. You can manually remove them to free up space.

### 1.4 Process CATH Domains

Run the following command to process the CATH domains:

```bash
python scripts/cath_processor.py --output_path BASE_DATA_DIR/cath_domain_boundaries.json
```

This script will:

- Parse the CATH domain boundaries file into a nested dictionary structure
- Save the parsed data to a JSON file


## 3. Download Model Checkpoint

Download the model checkpoint and note the file path where it is stored:

```bash
wget https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt
```

