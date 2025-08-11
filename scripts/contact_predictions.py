from __future__ import annotations
from pathlib import Path
import json
import biotite.structure as bs
from tqdm import tqdm  # progress bars
import biotite.structure as structure
import biotite.structure.io as io
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import numpy as np

# Project paths (scripts/ -> mint/ -> workspace/)
BASE_DATA_DIR = Path(__file__).parent.parent.parent / "DATA"
BASE_DIR = Path(__file__).parent.parent

def load_structure(input_path, hetero: bool = False, model: int = 1) -> structure.AtomArray | None: # altered, on contact-prediction branch
    """
    Load a structure and return a single-model AtomArray (protein-only if hetero=False).
    Accepts a local path or a 4-char PDB ID. Handles AtomArrayStack gracefully.
    """
    try:
        s = str(input_path)
        p = Path(s)

        if p.exists():
            atom = io.load_structure(s)  # may be AtomArray or AtomArrayStack
        else:
            if len(s) == 4 and s.isalnum():
                cif_file_object = rcsb.fetch(s, "cif", target_path=None)
                cif_file = pdbx.CIFFile.read(cif_file_object)
                atom = pdbx.get_structure(cif_file, model=model)  # already single model
            else:
                print(f"Error: CIF file not found at {s}.")
                return None

        # If it's a stack (multi-model), pick one model
        if isinstance(atom, structure.AtomArrayStack):
            m = max(1, model) - 1
            if m >= atom.stack_depth():
                m = 0
            atom = atom[m]  # -> AtomArray

        if atom is None or atom.array_length == 0:
            print(f"Error: No atoms loaded for structure from {s}.")
            return None

        # Now safe: atom is AtomArray; hetero is 1-D of length n_atoms
        return atom[atom.hetero == hetero]
    except Exception as e:
        print(f"Error loading structure {input_path}: {e}")
        return None

def locate_cif_file(pdb_id: str, features_file=BASE_DATA_DIR / "pdb_features.json") -> str | None:
    """Return path to local mmCIF file for *pdb_id* using pdb_features.json."""
    if not isinstance(features_file, Path):
        features_file = Path(features_file)

    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    with open(features_file, "r") as f:
        pdb_features = json.load(f)  # list of dicts

    for entry in pdb_features:
        if entry.get("pdb_id", "").upper() == pdb_id.upper():
            return entry.get("filepath")  # leave as string
    
    return None


def get_ca_atoms(
    atom_array: bs.AtomArray, chain_id: str, start: int, end: int
) -> bs.AtomArray:
    """
    Return only the Cα atoms from a given chain and residue range (inclusive).
    """
    if atom_array is None or atom_array.array_length == 0:
        raise ValueError("Empty AtomArray provided.")

    mask = (
        (atom_array.atom_name == "CA")
        & (atom_array.chain_id == chain_id)
        & (atom_array.res_id >= int(start))
        & (atom_array.res_id <= int(end))
        & (atom_array.ins_code == "")
    )
    return atom_array[mask]

def build_contact_mask(coord_lists, cutoff=5.0):
    """
    coord_lists: list of arrays/lists, each of shape (Li, 3)
    cutoff: distance threshold

    Returns
    -------
    mask: (N, N) int8 array with values:
          -1 if i,j from same list
           1 if from different lists AND distance < cutoff
           0 otherwise
    """
    # Concatenate all coordinates
    coords = np.vstack([np.asarray(c, dtype=float) for c in coord_lists])  # (N, 3)

    # Build an array of segment ids (which list each point came from)
    seg_ids = np.concatenate([
        np.full(len(c), k, dtype=np.int32) for k, c in enumerate(coord_lists)
    ])  # (N,)

    N = coords.shape[0]

    # Pairwise distances (broadcasted)
    diffs = coords[:, None, :] - coords[None, :, :]          # (N, N, 3)
    dists = np.linalg.norm(diffs, axis=-1)                    # (N, N)

    # Same-list mask
    same_list = (seg_ids[:, None] == seg_ids[None, :])        # (N, N)

    # Build final mask
    mask = np.zeros((N, N), dtype=np.int8)
    mask[same_list] = -1
    # For different lists, mark 1 where distance < cutoff
    np.fill_diagonal(dists, np.inf)  # avoid diagonal being < cutoff by accident
    mask[~same_list & (dists < cutoff)] = 1
    return mask

def add_contact_predictions_to_cath(
    dataset_root= BASE_DATA_DIR,
    cath_in: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """
    Build a JSON with CA coordinates for chains that have >1 domain and all domains are single-segment.

    - Reads: {dataset_root}/cath_domain_boundaries.json (or cath_in if provided)
    - Searches CIFs under: {dataset_root}/cif_unzipped
    - Writes: {dataset_root}/cath_ca_coords.json (or out_path if provided)
    """
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    cath_in_path = Path(cath_in) if cath_in else dataset_root / "cath_domain_boundaries.json"
    out_path = Path(out_path) if out_path else dataset_root / "cath_ca_coords.json"
    cif_root = dataset_root / "cif_unzipped"

    with open(cath_in_path, "r") as f:
        cath_dict = json.load(f)

    result: dict[str, dict[str, list[dict]]] = {}

    count = 0
    skipped_chains = 0

    # Wrap outer dict iteration
    for pdb_id, chains in tqdm(cath_dict.items(), desc="PDBs"):
        for chain_id, domains in tqdm(chains.items(), desc=f"{pdb_id} chains", leave=False):
            count += 1
            # Require >1 domain and exactly one segment per domain
            if len(domains) <= 1:
                skipped_chains += 1
                continue
            if not all(len(d.get("segments", [])) == 1 for d in domains):
                skipped_chains += 1
                continue

            cif_path = locate_cif_file(pdb_id=pdb_id)
            if cif_path is None:
                skipped_chains += 1
                continue

            
            struct = load_structure(BASE_DIR / cif_path)
            if struct is None or struct.array_length == 0:
                skipped_chains += 1
                continue
        
            output: list[dict] = []
            chain_ok = True

            # Validate every domain; if any fails, skip entire chain
            for d in domains:
                seg = d["segments"][0]
                start = int(seg["start"])
                end = int(seg["end"])

                ca = get_ca_atoms(struct, chain_id, start, end)
                coords = ca.coord.tolist() if ca.array_length() > 0 else []

                if len(coords) != (end - start) + 1:
                    chain_ok = False
                    break

                output.append(
                    {
                        "domain_idx": d["domain_idx"],
                        "segments": [
                            {
                                "segment_idx": seg["segment_idx"],
                                "start": start,
                                "end": end,
                                "c_alpha_coords": coords,
                            }
                        ],
                    }
                )

            if chain_ok and output: # create contact predictions mask
                list_of_domains = []
                for domain in output:
                    list_of_domains.append(domain["segments"][0]["c_alpha_coords"])
                contact_predictions = build_contact_mask(list_of_domains)
                output.append(contact_predictions)
                result.setdefault(pdb_id, {})[chain_id] = output
            else:
                skipped_chains += 1
                continue
        if count == 100:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f'The number of skipped chains is: {skipped_chains}')
    return out_path


add_contact_predictions_to_cath()
