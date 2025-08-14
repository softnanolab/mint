from pathlib import Path
import json 
from scripts.cath_processor import process_cath_domains

# Get the path to the test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


def test_process_cath_domains(
    test_input_path=TEST_DATA_DIR / "raw" / "test_cath_domain_boundaries.txt",
    test_output_path=TEST_DATA_DIR / "processed" / "test_processed_domain_boundaries.json",
):
    # Check that process_cath_domains parses the cath database correctly
    process_cath_domains(input_path=test_input_path, output_path=test_output_path)
    output_file = test_output_path

    # Basic file checks
    assert output_file.exists(), f"{output_file} does not exist"
    assert output_file.stat().st_size > 0, f"{output_file} is empty"

    # Load results
    with open(output_file, "r") as f:
        data_dict = json.load(f)

    # Expected structure and values with chain IDs
    expected = {
        "101m": {
            "A": [
                {"domain_idx": 1, "segments": [{"segment_idx": 1, "start": 0, "end": 153}]},
            ]
        },
        "102l": {
            "B": [
                {"domain_idx": 1, "segments": [{"segment_idx": 1, "start": 1, "end": 162}]},
            ]
        },
        "10gs": {
            "C": [
                {"domain_idx": 1, "segments": [
                    {"segment_idx": 1, "start": 2,   "end": 78},
                    {"segment_idx": 2, "start": 187, "end": 208},
                ]},
                {"domain_idx": 2, "segments": [
                    {"segment_idx": 1, "start": 79,  "end": 186},
                ]},
            ]
        },
        "10mh": {
            "D": [
                {"domain_idx": 1, "segments": [
                    {"segment_idx": 1, "start": 1,   "end": 186},
                    {"segment_idx": 2, "start": 285, "end": 327},
                ]},
                {"domain_idx": 2, "segments": [
                    {"segment_idx": 1, "start": 187, "end": 284},
                ]},
            ]
        }
    }

    # Validate top-level PDB IDs
    assert set(data_dict.keys()) == set(expected.keys()), \
        f"PDB IDs mismatch: {set(data_dict.keys())} vs {set(expected.keys())}"

    # Validate each PDB and chain
    for pdb_id, chains in expected.items():
        # Check chain IDs
        assert set(data_dict[pdb_id].keys()) == set(chains.keys()), \
            f"Chains for {pdb_id} incorrect: {set(data_dict[pdb_id].keys())} vs {set(chains.keys())}"

        # Validate each chain
        for chain_id, domains in chains.items():
            actual_domains = data_dict[pdb_id][chain_id]
            assert len(actual_domains) == len(domains), \
                f"Number of domains for {pdb_id}.{chain_id} incorrect: {len(actual_domains)} vs {len(domains)}"

            # Check each domain
            for exp_dom, act_dom in zip(domains, actual_domains):
                assert act_dom["domain_idx"] == exp_dom["domain_idx"], \
                    f"Domain index for {pdb_id}.{chain_id} mismatch: {act_dom['domain_idx']} vs {exp_dom['domain_idx']}"
                assert len(act_dom["segments"]) == len(exp_dom["segments"]), \
                    f"Segments count for {pdb_id}.{chain_id} domain {act_dom['domain_idx']} incorrect: " \
                    f"{len(act_dom['segments'])} vs {len(exp_dom['segments'])}"

                # Check each segment
                for exp_seg, act_seg in zip(exp_dom["segments"], act_dom["segments"]):
                    assert act_seg["segment_idx"] == exp_seg["segment_idx"], \
                        f"Segment index for {pdb_id}.{chain_id} domain {act_dom['domain_idx']} mismatch: " \
                        f"{act_seg['segment_idx']} vs {exp_seg['segment_idx']}"
                    assert act_seg["start"] == exp_seg["start"], \
                        f"Start for {pdb_id}.{chain_id} domain {act_dom['domain_idx']} segment {act_seg['segment_idx']} mismatch: " \
                        f"{act_seg['start']} vs {exp_seg['start']}"
                    assert act_seg["end"] == exp_seg["end"], \
                        f"End for {pdb_id}.{chain_id} domain {act_dom['domain_idx']} segment {act_seg['segment_idx']} mismatch: " \
                        f"{act_seg['end']} vs {exp_seg['end']}"


def teardown_function(function):
    """Remove the output file after tests run."""
    output_path = TEST_DATA_DIR / "processed" / "test_processed_domain_boundaries.json"
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
