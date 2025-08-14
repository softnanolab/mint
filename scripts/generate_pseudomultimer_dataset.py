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

3. Contact Masks file (`training.contact_masks.pkl.gz`)
    Contains a dictionary: [<domain_id>: contact mask (L x L) numpy.ndarray]
        <domain_id> follows the convention: PDBid_ChainCharacter e.g 1A2P_A
        L = pseudomultimer length (sum of all lengths of pseudochains)
   
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
    """
    Load a macromolecular structure from the RCSB PDB or from a local CIF file.

    Args:
        input_str : str
            Either:
            - A 4-character PDB identifier (e.g., "1ABC"), in which case the
            corresponding structure will be fetched from the RCSB in mmCIF format.
            - A local filesystem path to a `.cif` file containing the structure.

        hetero : bool, optional
            If False (default), return only non-heteroatom records (main polymer chains).
            If True, return only heteroatoms (e.g., ligands, water, ions).

    Returns:

        structure.AtomArray or None
            An `AtomArray` containing the atomic coordinates and metadata for the
            specified structure. This is a Biotite object representing the list of
            atoms in the model, including fields like atom name, residue name, chain ID,
            coordinates, and more.
            Returns None if the file cannot be loaded or contains no atoms.

    Notes:

    The returned AtomArray can be indexed, sliced, and filtered to select atoms
    or residues. It originates from the [biotite.structure](https://www.biotite-python.org/)
    package.
    """
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
    """
    Extract amino acid sequences from each chain in a macromolecular structure.

    Args:
        struct : structure.AtomArray
            A Biotite `AtomArray` representing the atomic coordinates and metadata
            of a structure model. Must contain standard atom annotation fields such
            as `chain_id`, `atom_name`, and `res_name`.

    Returns:
        Dict[str, str]
            A dictionary mapping each chain ID (as a string) to its corresponding
            amino acid sequence in one-letter code. Only residues containing a
            CA (alpha carbon) atom are included in the sequence.

    Notes:
        - The function identifies unique chain IDs in the structure, then filters
        for CA atoms (carbon-alpha) to ensure one residue per position.
        - Residue names in three-letter code are converted to one-letter code
        using `ProteinSequence.convert_letter_3to1()`.
        - Non-standard residues not recognized by the converter may raise an
        error or be skipped, depending on the implementation of
        `convert_letter_3to1()`.
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


def slice_sequence_by_residue_range(
    struct: structure.AtomArray, chain_id: str, start_res: int, end_res: int
) -> str:
    """
    Extract a contiguous subsequence from a specific chain in a structure,
    defined by a range of residue IDs.

    Args:
        struct : structure.AtomArray
            A Biotite `AtomArray` representing the atomic coordinates and metadata
            of a structure model. Must contain standard annotation fields such as
            `chain_id`, `atom_name`, `res_id`, and `ins_code`.

        chain_id : str
            The identifier of the chain from which to extract the subsequence.
            Sometimes this item can be referred to as a chain character for multimers,
            e.g for the PDB ID: 1A2P, you may supply a chain character 'A', to denote
            chain 'A' in the multimer.

        start_res : int
            The starting residue ID (inclusive) of the subsequence.

        end_res : int
            The ending residue ID (inclusive) of the subsequence.

    Returns:
        str
            The amino acid subsequence, in one-letter code, spanning residues
            from `start_res` to `end_res` in the specified chain.
            Returns an empty string if no residues match the criteria.

    Notes:
        - Only residues containing a CA atom (carbon-alpha) are considered,
          ensuring one amino acid per residue position.
        - Inserted residues (`ins_code != ""`) are excluded.
        - Residue names are converted from three-letter to one-letter codes using
          `ProteinSequence.convert_letter_3to1()`.
        - If the specified range is partially or completely missing from the chain,
          only available residues in the range will be included in the output.
    """
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
    Retrieve the 3D coordinates of CA (carbon-alpha) atoms from a specified chain
    in a multimer, within a given residue range.

    Args:
        struct : structure.AtomArray
            A Biotite `AtomArray` containing atomic coordinates and metadata for
            a structure model. Must include standard fields such as `atom_name`,
            `chain_id`, `res_id`, and `ins_code`.

        chain_id : str
            The identifier of the chain from which to extract CA atom coordinates,
            in the multimer.

        start_res : int
            The starting residue ID (inclusive) of the selection.

        end_res : int
            The ending residue ID (inclusive) of the selection.

    Returns:
        List
            A list of `[x, y, z]` coordinate triplets (floats) for each selected
            CA atom. Returns an empty list if no matching atoms are found.

    Raises:
        ValueError:
            If the provided `struct` is `None` or contains no atoms.

    Notes:
        - Only CA atoms (carbon-alpha) are included.
        - Residues with non-empty insertion codes (`ins_code != ""`) are excluded.
        - The order of returned coordinates follows the order of atoms in the
          filtered `AtomArray`.
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
    # TODO: vectorise the computation in this function
    """
    Generate a binary contact map for a pseudomultimer, with intrachain residue pairs
    marked as ignored.

    Args:
        domain_coords : List[List[List[float]]]
            Nested list of 3D coordinates in the form:
            [
                [ [x1, y1, z1], [x2, y2, z2], ... ],  # chain 1 residues
                [ [x1, y1, z1], [x2, y2, z2], ... ],  # chain 2 residues
                ...
            ]
            Each inner list corresponds to one chain, and each coordinate
            triplet corresponds to the position of a residue's representative atom
            (e.g., CA).

        cutoff : float, optional
            Distance threshold in Angstroms for determining contacts. Defaults to 5.0.

    Returns:
        numpy.ndarray
            A 2D array of shape `(L, L)`, where `L` is the total number of residues
            in the pseudomultimer. Each entry is:
            - `-1` for intrachain residue pairs (ignored positions)
            - `1` if the interchain pair is in contact (Euclidean distance ≤ `cutoff`)
            - `0` if the interchain pair is not in contact (distance > `cutoff`)

    Notes:
        - Residues are indexed in sequence order across all chains; chain boundaries
          are tracked internally to assign `-1` for intrachain contacts.
        - The contact mask is symmetric by definition, though the computation
          iterates over all `(i, j)` pairs explicitly.
        - The function operates on the representative coordinate per residue
          provided in `domain_coords`; if these are CA coordinates, the resulting
          contact map is a CA-CA contact map.
    """
    sequence_length = 0
    domain_intervals = []
    for chain in domain_coords:
        sequence_length += len(chain)
        if not domain_intervals:
            domain_intervals.append([0, len(chain) - 1])
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
    """
    Locate the local path to an mmCIF structure file for a given PDB ID using
    metadata from a `pdb_features.json`-like structure.

    Args:
        pdb_id : str
            Four-character PDB identifier (case-insensitive) for the desired
            structure.

        pdb_features : List[Dict]
            A list of dictionaries containing metadata for available structures.
            Each dictionary is expected to have:
                - `"pdb_id"` : str
                - `"filepath"` : str (path relative to `base_dir`)
            Note: this is the information in pdb_features.json, loaded as a list
            of dictionaries.

        base_dir : str or Path, optional
            This is the project directory and assists with retrieving the path to the cif
            files. Defaults to the `PROJECT_ROOT` constant.

    Returns:
        str or None
            Absolute path to the requested mmCIF file if found, otherwise `None`.

    Notes:
        - The search is case-insensitive with respect to the PDB ID.
        - The `filepath` from the matching `pdb_features` entry is joined with
          `base_dir` to create the absolute path.
        - This function does not check whether the file actually exists on disk;
          it only constructs the path based on metadata.
    """
    for entry in pdb_features:
        if entry.get("pdb_id", "").upper() == pdb_id.upper():
            output_path = Path(str(base_dir) + "/" + entry.get("filepath")).resolve()
            return str(output_path)
    return None


def build_dataset(pdb_features: List[Dict], cath_data_path=CATH_JSON):
    """
    Construct sequence, domain link, and contact mask datasets from CATH domain
    annotations and corresponding structure files, for pseudomultimers. Pseudomultimers
    are chains in the PDB with only contiguous domains. Crucially, they must have more
    than one domain.

    Args:
        pdb_features : List[Dict]
            A list of metadata entries describing available structures, typically
            parsed from a `pdb_features.json` file. Each entry should include:
                - `"pdb_id"` : str
                - `"filepath"` : str (path relative to base directory)
            Note: this is the information in pdb_features.json, loaded as a list
            of dictionaries.

        cath_data_path : str or Path, optional
            Path to a JSON file containing CATH classification and domain segment
            information for PDB chains. Defaults to the `CATH_JSON` constant.

    Returns:
        tuple
            A 4-tuple containing:

            - **sequences** (`Dict[str, str]`): Maps `domain_id` → amino acid
              sequence in one-letter code. Domain IDs follow the format
              `"pdb_id_chain_id_index"`.

            - **links** (`List[List[str]]`): Each element is a list of domain IDs
              that belong to the same chain and should be considered together.
              Only chains with at least two domains are included.

            - **contact_masks** (`Dict[str, numpy.ndarray]`): Maps `chain_id`
              (formatted as `"pdb_id_chain_id"`) → 2D NumPy array contact mask
              of shape `(L, L)`, where `L` is the total number of residues in
              all domains for that pseudomultimer.
                  - `-1` marks intradomain residue pairs (ignored)
                  - `0` for interdomain pairs further than the cutoff
                  - `1` for interdomain pairs within the cutoff distance

            - **skipped_items** (`List[Dict[str, str]]`): Metadata for skipped
              domains/chains, including PDB ID, chain ID, and reason for skipping.

    Notes:
        - Sequences are extracted using `slice_sequence_by_residue_range()` and
          contact masks are computed from CA coordinates via `get_contact_mask()`.
        - The cutoff for contacts is fixed at 5.0 Å in the current implementation.
        - Structures are loaded from local CIF files using `locate_cif_file()`
          and `load_structure()`.
    """
    with open(cath_data_path) as f:
        cath_data = json.load(f)

    sequences: Dict[str, str] = {}
    links: List[List[str]] = []
    contact_masks: Dict[str : NDArray[np.int_]] = {}
    skipped_items: List[Dict[str, str]] = []

    total_entries = sum(len(chains) for chains in cath_data.values())

    with tqdm(total=total_entries, desc="Processing CATH entries") as pbar:
        for pdb_id, chains in islice(cath_data.items(), 1000):
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
                contact_mask = get_contact_mask(domain_coords, cutoff=8.0)
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
    """
    Generate training dataset files for MINT retraining from CATH domain data
    and local PDB structure files. The dataset includes pseudomultimers. These are
    chains in the PDB which have only contiguous domains, and crucially they must
    have more than two domains.

    This function:
      1. Loads PDB metadata from `pdb_features.json`.
      2. Builds domain sequences, inter-domain links, and residue contact masks
         using `build_dataset()`.
      3. Saves:
         - A gzipped `.seqs.txt.gz` file mapping domain IDs to sequences.
         - A gzipped `.links.txt.gz` file containing space-separated domain IDs
           for each chain with ≥2 domains.
         - A gzipped pickle file with a dictionary mapping chain IDs to contact
           mask matrices.
         - A `skipped_items.json` file listing domains/chains skipped during
           dataset creation.

    Args:
        links_output_path : str
            Path to write the `training.links.txt.gz` file.

        seqs_output_path : str
            Path to write the `training.seqs.txt.gz` file.

        contact_masks_ouput_path : str
            Path to write the `training.contact_masks.pkl.gz` pickle file.

        pdb_features_path : str or Path, optional
            Path to the `pdb_features.json` file containing PDB metadata.
            Defaults to `PDB_FEATURES`.

        cath_data_path : str or Path, optional
            Path to the CATH domain JSON file. Defaults to `CATH_JSON`.

    Raises:
        FileNotFoundError:
            If `pdb_features.json` cannot be found at `pdb_features_path`.

    Notes:
        - All output files are gzipped.
        - The contact mask matrices use:
            * `-1` for intra-domain pairs (ignored during loss computation)
            * `0` for inter-domain pairs with CA-CA distance > cutoff
            * `1` for inter-domain pairs with distance ≤ cutoff
        - Skipped items include missing CIF files, empty sequences, and
          structures that fail to load.
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
