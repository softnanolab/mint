"""Generate a list of PDB ids to download"""
import os

from dotenv import load_dotenv, find_dotenv
import fire
from rcsbsearchapi import rcsb_attributes as attrs

# Load environment variables from .env file
load_dotenv(find_dotenv())
DATA_DIR = os.getenv("DATA_DIR")


def get_filtered_pdb_ids(max_resolution: float, max_date: str, min_length: int) -> list:
    """
    Query RCSB PDB database with filters and return list of PDB IDs

    Args:
        max_resolution (float): Maximum resolution cutoff in Angstroms
        max_date (str): Maximum deposition date in YYYY-MM-DD format
        min_length (int): Minimum protein sequence length

    Returns:
        list: List of filtered PDB IDs that match the query criteria
    """
    # Create query terminals
    date_query = attrs.rcsb_accession_info.deposit_date < max_date
    resolution_query = attrs.rcsb_entry_info.resolution_combined <= max_resolution
    length_query = attrs.entity_poly.rcsb_sample_sequence_length > min_length
    protein_query = attrs.entity_poly.rcsb_entity_polymer_type == "Protein"

    # Combine queries
    query = date_query & resolution_query & length_query & protein_query

    # Execute query and collect results
    results = query()
    return list(results)


def main(
    base_data_dir: str = DATA_DIR,
    resolution: float = 9.0,
    date: str = "2020-05-01",
    length: int = 20,
) -> None:
    """
    Filter PDB IDs based on specified criteria and save them to a file

    Args:
        base_data_dir (str): Base directory path where data will be stored.
            defaults to DATA_DIR from .env file
        resolution (float): Maximum resolution cutoff in Angstroms.
            defaults to 9.0
        date (str): Maximum deposition date in YYYY-MM-DD format.
            defaults to "2020-05-01"
        length (int): Minimum protein sequence length.
            defaults to 20
    """
    assert os.path.exists(base_data_dir), f"Base data directory {base_data_dir} does not exist"

    # Get filtered PDB IDs
    pdb_ids = get_filtered_pdb_ids(max_resolution=resolution, max_date=date, min_length=length)

    # Save results
    output_file_path = os.path.join(base_data_dir, "raw", "pdb_ids.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Write PDB IDs to file as comma-separated values with newlines
    with open(output_file_path, "w") as f:
        f.write(",".join(pdb_ids))

    print(f"Saved {len(pdb_ids)} PDB IDs to {output_file_path}")


if __name__ == "__main__":
    # Example usage:
    # python scripts/generate_pdb_ids.py --base_data_dir /path/to/data --resolution 9.0 --date 2020-05-01 --length 20
    fire.Fire(main)
