from __future__ import annotations
from itertools import islice

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
from numpy.typing import NDArray
import numpy as np

import numpy as np
from tqdm import tqdm
import gzip
import pickle
import fire

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

CATH_JSON = PROJECT_ROOT / "resources/cath_domain_boundaries.json"

DATA_DIR = PROJECT_ROOT.parent / "DATA"

PDB_FEATURES = DATA_DIR / "pdb_features.json"

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
        seq = "".join(
            ProteinSequence.convert_letter_3to1(res.res_name) for res in chain_struct
        )
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
    return "".join(
        ProteinSequence.convert_letter_3to1(res.res_name) for res in chain_struct
    )


def get_ca_atoms(struct: structure.AtomArray, chain_id: str, start_res: int, end_res: int) -> List:
    """
    Returns the 3D coordinates of just the Cα atoms from a given chain
    and residue range (inclusive).
    """
    if struct is None or struct.array_length == 0:
        raise ValueError("Empty AtomArray provided.")

    mask = (
        (struct.atom_name == "CA")
        & (struct.chain_id == chain_id)
        & (struct.res_id >= start_res)
        & (struct.res_id <= end_res)
        & (struct.ins_code == "")
    )
    struct = struct[mask]
    return struct.coord.tolist() if struct.array_length() > 0 else []


def get_contact_mask(domain_coords: List[List[List]], cutoff=5.0):
    sequence_length = 0
    domain_intervals = []
    for chain in domain_coords:
        sequence_length += len(chain)
        if not domain_intervals:
            domain_intervals.append([1, len(chain)])
        else:
            domain_intervals.append(
                [domain_intervals[-1][-1] + 1, domain_intervals[-1][-1] + len(chain)]
            )

    contact_mask = np.zeros((sequence_length, sequence_length), dtype=int)
    for start, end in domain_intervals:
        contact_mask[start : end + 1, start : end + 1] = -1  # ignore intrachain residue contacts

    # domain_coords: List[List[List[float]]]
    coords = np.stack(
        [np.asarray(pt, dtype=float) for chain in domain_coords for pt in chain], axis=0
    )  # shape: (N, 3)

    N = contact_mask.shape[0]
    for i in range(N):
        for j in range(N):
            if contact_mask[i, j] == -1:
                continue
            contact_mask[i, j] = 0 if np.linalg.norm(coords[i] - coords[j]) > 5.0 else 1

    return contact_mask


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def locate_cif_file(pdb_id: str, pdb_features: List[Dict], base_dir=PROJECT_ROOT) -> str | None:
    """Return path to local mmCIF file for *pdb_id* using pdb_features.json."""
    for entry in pdb_features:
        if entry.get("pdb_id", "").upper() == pdb_id.upper():
            output_path = Path(str(base_dir) + "/" + entry.get("filepath")).resolve()
            return str(output_path)
    return None


def build_dataset(pdb_features: List[Dict], cath_data_path=CATH_JSON):
    """Generate sequence and link data structures.

    Returns:
        sequences (Dict[str, str]): domainID -> sequence
        links (List[Tuple[str, str]]): list of domainID pairs (space-separated later)
        contact_mask (Dict[str, NDArray]): chain_id -> contact mask for residues in the chain
    """
    with open(cath_data_path) as f:
        cath_data = json.load(f)

    sequences: Dict[str, str] = {}
    links: List[List[str]] = []
    contact_masks: Dict[str : NDArray[np.int_]] = {}
    skipped_items: List[Dict[str, str]] = []

    total_entries = sum(len(chains) for chains in cath_data.values())

    with tqdm(total=total_entries, desc="Processing CATH entries") as pbar:
        for pdb_id, chains in islice(cath_data.items(), 100):
            for chain_id, domains in chains.items():
                pbar.update(1)

                # require >1 domain and each with exactly 1 segment
                if len(domains) <= 1:
                    continue
                if not all(len(domain["segments"]) == 1 for domain in domains):
                    continue

                cif_path = locate_cif_file(pdb_id, pdb_features)
                if cif_path is None:
                    skipped_items.append({
                        "pdb_id": pdb_id,
                        "chain_id": chain_id,
                        "reason": "cif_not_found"
                    })
                    continue

                struct = load_structure(cif_path)
                if struct is None:
                    skipped_items.append({
                        "pdb_id": pdb_id,
                        "chain_id": chain_id,
                        "reason": "structure_load_failed"
                    })
                    continue

                # build the contact mask here by doing contact_mask(domains)
                # contact_mask = create_contact_mask(struct)

                domain_ids: List[str] = []
                domain_coords: List[List] = []
                for idx, domain in enumerate(domains, start=1):

                    seg = domain["segments"][0]
                    seq = slice_sequence_by_residue_range(struct, chain_id, seg["start"], seg["end"])

                    if not seq:
                        skipped_items.append({
                            "pdb_id": pdb_id,
                            "chain_id": chain_id,
                            "reason": "empty_sequence",
                            "domain_idx": idx
                        })
                        domain_ids = []
                        break

                    domain_coords.append(get_ca_atoms(struct, chain_id, seg["start"], seg["end"]))

                    dom_id = f"{pdb_id}_{chain_id}_{idx}"
                    sequences[dom_id] = seq
                    domain_ids.append(dom_id)

                # find the contact mask for this chain (pseudochains are domains)
                contact_mask = get_contact_mask(domain_coords, cutoff=5.0)
                chain_id = f"{pdb_id}_{chain_id}"
                contact_masks[chain_id] = contact_mask

                # add whole domain group line if >=2 domains
                if len(domain_ids) >= 2:
                    links.append(domain_ids)

    return sequences, links, contact_masks, skipped_items


def main(
    links_output_path: str,
    seqs_output_path: str,
    contact_masks_ouput_path: str,
    pdb_features_path=PDB_FEATURES,
    cath_data_path=CATH_JSON,
):
    """Generate gzipped links and sequences files for MINT retraining, along with
    a pickle file containing the contact masks for each chain""

    Args:
        dataset_path: Directory containing mmCIF files.
        links_output_path: Path to write the `training.links.txt.gz` file.
        seqs_output_path: Path to write the `training.seqs.txt.gz` file.
        contact_masks_output_path: Path to write the 'training.contact_masks.
    """
    # Load pdb_features.json
    if not pdb_features_path.exists():
        raise FileNotFoundError(f"pdb_features.json not found at {pdb_features_path}")

    with open(pdb_features_path) as f:
        pdb_features = json.load(f)

    print(f"Links output : {links_output_path}")
    print(f"Seqs  output : {seqs_output_path}")
    print(f"Contact masks output: {contact_masks_ouput_path}")
    print(f"Loaded {len(pdb_features)} PDB entries from pdb_features.json")

    sequences, links, contact_masks, skipped_items = build_dataset(
        pdb_features, cath_data_path=cath_data_path
    )

    # Write sequences file
    with gzip.open(seqs_output_path, "wt") as f_seq:
        for dom_id, seq in sequences.items():
            f_seq.write(f"{dom_id} {seq}\n")

    # Write links file
    with gzip.open(links_output_path, "wt") as f_link:
        for ids in links:
            f_link.write(" ".join(ids) + "\n")

    # Write the contact_masks file
    with gzip.open(contact_masks_ouput_path, "wb") as f_contact_mask:
        pickle.dump(contact_masks, f_contact_mask, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Wrote {len(sequences)} sequences, {len(links)} links, and {len(contact_masks)} contact masks."
    )
    print(f"Skipped {len(skipped_items)} items.")

    # Save skipped items to JSON file
    skipped_output_path = Path(links_output_path).parent / "skipped_items.json"
    with open(skipped_output_path, "w") as f:
        json.dump(skipped_items, f, indent=2)
    print(f"Skipped items saved to: {skipped_output_path}")


if __name__ == "__main__":
    fire.Fire(main)
