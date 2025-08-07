from pathlib import Path
from typing import Dict, List, Any
import json

BASE_DIR = Path(__file__).parent.parent

class CATHProcessor:
    def __init__(self) -> None:
        """
        Initialize the CATHProcessor.
        """
        self.database_path = BASE_DIR / "resources/cath_domain_boundaries.txt"
        assert self.database_path.exists(), f"Database file not found at {self.database_path}"
        

    def main(self, output_path: str = BASE_DIR / "resources/cath_domain_boundaries.json") -> None:
        """Parses the CathDomall file into a nested dictionary structure.

        Args:
            output_path (str): The path to save the parsed data.

        Returns:
            None
        """
        result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        with open(self.database_path, "r") as f:
            for line_num, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                tokens = line.split()
                if len(tokens) < 3:
                    # Malformed – skip
                    continue

                chain_name = tokens[0]  # e.g. 1chmA
                pdb_id, chain_id = chain_name[:4], chain_name[4]

                num_domains = int(tokens[1][1:])  # remove leading 'D'
                # fragments = int(tokens[2][1:])  # 'Fxx' – not used here

                idx = 3  # pointer into tokens after Dxx Fxx
                for domain_idx in range(1, num_domains + 1):
                    if idx >= len(tokens):
                        break  # safety
                    num_segments = int(tokens[idx])
                    idx += 1

                    segments = []
                    for segment_idx in range(1, num_segments + 1):
                        if idx + 5 >= len(tokens):
                            break  # malformed – stop processing
                        # token pattern: chain start_i insert_start chain end_i insert_end
                        _chain_start = tokens[idx]; start_res = int(tokens[idx + 1]); _insert_start = tokens[idx + 2]
                        _chain_end = tokens[idx + 3]; end_res = int(tokens[idx + 4]); _insert_end = tokens[idx + 5]
                        segments.append({
                            "segment_idx": segment_idx,
                            "start": start_res,
                            "end": end_res,
                        })
                        idx += 6

                    result.setdefault(pdb_id, {}).setdefault(chain_id, []).append({
                        "domain_idx": domain_idx,
                        "segments": segments,
                    })

        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)

