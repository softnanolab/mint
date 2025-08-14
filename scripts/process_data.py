#!/usr/bin/env python3

import os
import gzip
import json
import math
import multiprocessing as mp
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from Bio import PDB
from tqdm import tqdm
import fire
import numpy as np
from dotenv import load_dotenv, find_dotenv

# Biotite imports for structure handling
import biotite.structure as structure
import biotite.structure.io as io
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
from biotite.sequence import ProteinSequence

# Load environment variables from .env file
load_dotenv(find_dotenv())
DATA_DIR = os.getenv("DATA_DIR")


def get_structure_files(cif_zipped_dir: Path) -> Dict[str, List[Path]]:
    """Get all .pdb.gz and .cif.gz files from source directory.

    Args:
        cif_zipped_dir: Directory containing gzipped files

    Returns:
        Dictionary with file types as keys and lists of file paths as values
    """
    cif_zipped_dir = Path(cif_zipped_dir)
    files = {
        "pdb": list(cif_zipped_dir.glob("*.pdb.gz")),
        "cif": list(cif_zipped_dir.glob("*.cif.gz")),
    }

    total_files = sum(len(files[k]) for k in files)
    print(
        f"Found {len(files['pdb'])} PDB and {len(files['cif'])} CIF files (total: {total_files})"
    )
    return files


def create_folder_structure(
    base_data_dir: Path, total_files: int, files_per_folder: int
) -> Dict[int, Path]:
    """Create numbered folders for organizing structure files.

    Args:
        base_data_dir: Base directory
        total_files: Total number of files to process
        files_per_folder: Number of files per subfolder

    Returns:
        Dictionary mapping folder numbers to folder paths
    """
    unzipped_dir = base_data_dir / "raw" / "cif_unzipped"
    unzipped_dir.mkdir(parents=True, exist_ok=True)

    num_folders = math.ceil(total_files / files_per_folder)
    folders = {}

    print(f"Creating {num_folders} folders...")
    for i in range(num_folders):
        folder_name = f"{i:03d}"
        folder_path = unzipped_dir / folder_name
        folder_path.mkdir(exist_ok=True)
        folders[i] = folder_path

    return folders


def is_amino_acid(residue: PDB.Residue.Residue) -> bool:
    """Check if a residue is a standard amino acid.

    Args:
        residue: Residue object to check

    Returns:
        True if residue is a standard amino acid, False otherwise
    """
    return residue.get_resname() in PDB.Polypeptide.standard_aa_names


def is_ligand_chain(chain: PDB.Chain.Chain) -> bool:
    """Check if a chain is a ligand chain (contains no amino acids).

    Args:
        chain: Chain object to check

    Returns:
        True if chain contains no amino acids, False otherwise
    """
    return all(not is_amino_acid(residue) for residue in chain.get_residues())


def load_structure(input_str: str, hetero: bool = False) -> structure.AtomArray | None:
    """Load structure using biotite.
    
    Args:
        input_str: PDB ID or file path
        hetero: Whether to include hetero atoms
        
    Returns:
        AtomArray or None if loading failed
    """
    try:
        # PDB ID
        if len(input_str) == 4 and input_str.isalnum():
            cif_file_object = rcsb.fetch(input_str, "cif", target_path=None)
            cif_file = pdbx.CIFFile.read(cif_file_object)
            atom_array = pdbx.get_structure(cif_file, model=1)
        else:  # local path
            if not Path(input_str).exists():
                print(f"Error: CIF file not found at {input_str}.")
                return None
            atom_array = io.load_structure(input_str)

        if atom_array is None or atom_array.array_length == 0:
            print(f"Error: No atoms loaded for structure from {input_str}.")
            return None
        return atom_array[atom_array.hetero == hetero]
    except Exception as e:
        print(f"Error loading structure {input_str}: {e}")
        return None


def get_sequence(struct: structure.AtomArray) -> Dict[str, str]:
    """Extract sequences from structure using biotite.
    
    Args:
        struct: AtomArray structure
        
    Returns:
        Dictionary mapping chain IDs to sequences
    """
    sequences: Dict[str, str] = {}
    for chain_id in np.unique(struct.chain_id):
        chain_mask = (struct.chain_id == chain_id) & (struct.atom_name == "CA")
        chain_struct = struct[chain_mask]
        seq = "".join(
            ProteinSequence.convert_letter_3to1(res.res_name) for res in chain_struct
        )
        sequences[str(chain_id)] = seq
    return sequences


def analyze_structure(
    fpath: str, pdb_id: str
) -> Dict[str, Union[str, int, Dict[str, int]]]:
    """Analyze a structure to extract key information using biotite.

    Args:
        fpath: Path to the structure file
        pdb_id: PDB ID

    Returns:
        Dictionary containing analysis results
    """
    # Load structure using biotite
    struct = load_structure(fpath, hetero=False)
    if struct is None:
        return None

    # Get sequences for all chains
    sequences = get_sequence(struct)
    if not sequences:
        return None

    chain_lengths = {chain_id: len(seq) for chain_id, seq in sequences.items()}
    total_residues = sum(chain_lengths.values())
    num_chains = len(chain_lengths)

    # Try to get date from file metadata (simplified approach)
    try:
        # Use file modification time as fallback
        date = int(Path(fpath).stat().st_mtime)
    except:
        date = 0

    return {
        "filepath": str(fpath),
        "pdb_id": pdb_id,
        "date": date,
        "structure_type": (
            "Monomer" if num_chains == 1 else f"Multimer ({num_chains}-mer)"
        ),
        "num_chains": num_chains,
        "chain_lengths": chain_lengths,
        "total_residues": total_residues,
    }


def process_file(args: Tuple[Path, Path, int]) -> Tuple[bool, Optional[Dict]]:
    """Process a single structure file: unzip and analyze.

    Args:
        args: Tuple containing (gz_file, target_folder, file_index)

    Returns:
        Tuple containing:
            - Success status (bool)
            - Analysis results (Dict) or None if failed
    """
    gz_file, target_folder, _ = args
    try:
        output_file = target_folder / gz_file.stem

        # Unzip file
        with gzip.open(gz_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        pdb_id = output_file.stem.replace(".pdb", "").replace(".cif", "")
        results = analyze_structure(fpath=output_file, pdb_id=pdb_id)
        if results is None:
            return False, None
        else:
            return True, results
    except Exception as e:
        print(f"Error processing {gz_file.name}: {str(e)}")
        return False, None


def main(base_data_dir: str = DATA_DIR, files_per_folder: int = 1000, num_cpus: int = 8) -> None:
    """Process structure files: unzip, organize into folders, and analyze.

    Args:
        base_data_dir: Base directory containing the data
        files_per_folder: Number of files per subfolder
        num_cpus: Number of CPU cores to use for parallel processing
    """
    base_data_dir = Path(base_data_dir)
    if not base_data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_data_dir}")

    # Get all structure files
    structure_files = get_structure_files(base_data_dir / "raw" / "cif_zipped")
    all_files = structure_files["pdb"] + structure_files["cif"]
    total_files = len(all_files)

    if total_files == 0:
        print("No structure files found!")
        return

    # Create folder structure
    folders = create_folder_structure(base_data_dir, total_files, files_per_folder)

    # Prepare processing tasks
    tasks = []
    for i, gz_file in enumerate(all_files):
        folder_num = i // files_per_folder
        target_folder = folders[folder_num]
        tasks.append((gz_file, target_folder, i))

    # Process files in parallel
    num_cpus = min(num_cpus, mp.cpu_count(), len(tasks))
    print(f"\nProcessing {total_files} files using {num_cpus} CPU cores...")

    results = []
    failed_files = []

    with mp.Pool(num_cpus) as pool:
        for success, result in tqdm(
            pool.imap_unordered(process_file, tasks),
            total=len(tasks),
            desc="Processing files",
        ):
            if success and result:
                if result["num_chains"] > 0 and result["total_residues"] > 0:
                    results.append(result)
                else:
                    print(
                        f"Warning: Skipping {result['pdb_id']} due to 0 chains or residues"
                    )
            else:
                failed_files.append(str(tasks[len(results) + len(failed_files)][0]))

    # Save results
    save_path = base_data_dir / "pdb_features.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {save_path}")

    failed_path = base_data_dir / "failed_files.json"
    with open(failed_path, "w") as f:
        json.dump(failed_files, f, indent=4)
    print(f"Failed files list saved to: {failed_path}")

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(results)}/{total_files} files")
    print(f"Failed to process: {len(failed_files)} files")


if __name__ == "__main__":
    fire.Fire(main)
