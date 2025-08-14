from __future__ import annotations

"""Generate pseudo-multimer dataset files for MINT retraining from CATH domain boundaries.

Each eligible CATH chain (with more than one domain, and each domain consisting of exactly one segment)
becomes one pseudo-multimer. Every domain acts as a pseudo-monomer (pseudo-chain).

This script produces two gzipped text files:

1. Sequences file (`training.seqs.txt.gz`):
    Each line contains a domain ID and its amino acid sequence, separated by a space:
        <domain_id> <sequence>
    Example:
        1abc_A_1 MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPQ

2. Links file (`training.links.txt.gz`):
    Each line contains the space-separated list of domain IDs that form a pseudo-multimer:
        <domain_id_1> <domain_id_2> ... <domain_id_N>
    Example:
        1abc_A_1 1abc_A_2

Notes
-----
* Residue ranges are interpreted using PDB residue numbers.
* Only protein backbone atoms (hetero == False) are considered when building sequences.
* Structures are read from locally-downloaded mmCIF files residing under
  DATASET_ROOT (searched recursively).
"""

import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm
import gzip
import fire

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

CATH_JSON = PROJECT_ROOT / "resources/cath_domain_boundaries.json"

# -----------------------------------------------------------------------------
# Utility wrappers (copied from user-provided snippet)
# -----------------------------------------------------------------------------

import biotite.structure as structure
import biotite.structure.io as io
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
from biotite.sequence import ProteinSequence


def load_structure(input_str: str, hetero: bool = False) -> structure.AtomArray | None:
    try:
        # PDB ID
        if len(input_str) == 4 and input_str.isalnum():
            cif_file_object = rcsb.fetch(input_str, "cif", target_path=None)
            cif_file = pdbx.CIFFile.read(cif_file_object)
            atom_array = pdbx.get_structure(cif_file, model=1)
        else:  # local path
            if not os.path.exists(input_str):
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
    sequences: Dict[str, str] = {}
    for chain_id in np.unique(struct.chain_id):
        chain_mask = (struct.chain_id == chain_id) & (struct.atom_name == "CA")
        chain_struct = struct[chain_mask]
        seq = "".join(ProteinSequence.convert_letter_3to1(res.res_name) for res in chain_struct)
        sequences[str(chain_id)] = seq
    return sequences


def slice_sequence_by_residue_range(
    struct: structure.AtomArray, chain_id: str, start_res: int, end_res: int
) -> str:
    """Return 1-letter sequence between start_res and end_res inclusive."""
    mask = (
        (struct.chain_id == chain_id)
        & (struct.atom_name == "CA")
        & (struct.res_id >= start_res)
        & (struct.res_id <= end_res)
        & (struct.ins_code == "")
    )
    if not np.any(mask):
        return ""
    chain_struct = struct[mask]
    return "".join(ProteinSequence.convert_letter_3to1(res.res_name) for res in chain_struct)


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def locate_cif_file(pdb_id: str, dataset_root: Path, pdb_features: List[Dict]) -> str | None:
    """Return path to local mmCIF file for *pdb_id* using pdb_features.json."""
    for entry in pdb_features:
        if entry.get("pdb_id", "").upper() == pdb_id.upper():
            return entry.get("filepath")
    return None


def build_dataset(dataset_root: Path, pdb_features: List[Dict]):
    """Generate sequence and link data structures.

    Returns:
        sequences (Dict[str, str]): domainID -> sequence
        links (List[Tuple[str, str]]): list of domainID pairs (space-separated later)
    """
    with open(CATH_JSON) as f:
        cath_data = json.load(f)

    sequences: Dict[str, str] = {}
    links: List[List[str]] = []
    skipped_items: List[Dict[str, str]] = []

    total_entries = sum(len(chains) for chains in cath_data.values())

    with tqdm(total=total_entries, desc="Processing CATH entries") as pbar:
        for pdb_id, chains in cath_data.items():
            for chain_id, domains in chains.items():
                pbar.update(1)

                # require >1 domain and each with exactly 1 segment
                if len(domains) <= 1:
                    continue
                if not all(len(domain["segments"]) == 1 for domain in domains):
                    continue

                cif_path = locate_cif_file(pdb_id, dataset_root, pdb_features)
                if cif_path is None:
                    skipped_items.append(
                        {"pdb_id": pdb_id, "chain_id": chain_id, "reason": "cif_not_found"}
                    )
                    continue

                struct = load_structure(cif_path)
                if struct is None:
                    skipped_items.append(
                        {"pdb_id": pdb_id, "chain_id": chain_id, "reason": "structure_load_failed"}
                    )
                    continue

                domain_ids: List[str] = []
                for idx, domain in enumerate(domains, start=1):
                    seg = domain["segments"][0]
                    seq = slice_sequence_by_residue_range(
                        struct, chain_id, seg["start"], seg["end"]
                    )
                    if not seq:
                        skipped_items.append(
                            {
                                "pdb_id": pdb_id,
                                "chain_id": chain_id,
                                "reason": "empty_sequence",
                                "domain_idx": idx,
                            }
                        )
                        domain_ids = []
                        break
                    dom_id = f"{pdb_id}_{chain_id}_{idx}"
                    sequences[dom_id] = seq
                    domain_ids.append(dom_id)

                # add whole domain group line if >=2 domains
                if len(domain_ids) >= 2:
                    links.append(domain_ids)

    return sequences, links, skipped_items


def main(
    dataset_path: str,
    links_output_path: str,
    seqs_output_path: str,
):
    """Generate gzipped links and sequences files for MINT retraining.

    Args:
        dataset_path: Directory containing mmCIF files.
        links_output_path: Path to write the `training.links.txt.gz` file.
        seqs_output_path: Path to write the `training.seqs.txt.gz` file.
    """
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    # Load pdb_features.json
    pdb_features_path = dataset_root.parent / "pdb_features.json"
    if not pdb_features_path.exists():
        raise FileNotFoundError(f"pdb_features.json not found at {pdb_features_path}")

    with open(pdb_features_path) as f:
        pdb_features = json.load(f)

    print(f"Dataset root : {dataset_root}")
    print(f"Links output : {links_output_path}")
    print(f"Seqs  output : {seqs_output_path}")
    print(f"Loaded {len(pdb_features)} PDB entries from pdb_features.json")

    sequences, links, skipped_items = build_dataset(dataset_root, pdb_features)

    # Write sequences file
    with gzip.open(seqs_output_path, "wt") as f_seq:
        for dom_id, seq in sequences.items():
            f_seq.write(f"{dom_id} {seq}\n")

    # Write links file
    with gzip.open(links_output_path, "wt") as f_link:
        for ids in links:
            f_link.write(" ".join(ids) + "\n")

    print(f"Wrote {len(sequences)} sequences and {len(links)} links.")
    print(f"Skipped {len(skipped_items)} items.")

    # Save skipped items to JSON file
    skipped_output_path = Path(links_output_path).parent / "skipped_items.json"
    with open(skipped_output_path, "w") as f:
        json.dump(skipped_items, f, indent=2)
    print(f"Skipped items saved to: {skipped_output_path}")


if __name__ == "__main__":
    fire.Fire(main)
