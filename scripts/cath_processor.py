from pathlib import Path
from typing import Dict, List, Any
import json
import fire

BASE_DIR = Path(__file__).parent.parent


def process_cath_domains(
    input_path=BASE_DIR / "resources/cath_domain_boundaries.txt", output_path: str = None
) -> None:
    """Parses the CathDomall file into a nested dictionary structure which looks like:

    {
    "pdb_id" : {
                "chain_id" : [
                             {
                             "domain_idx" : 1
                             "segments" : [
                                         {
                                         "segment_idx" : 1
                                         "start" : starting residue index
                                         "end" : ending residue index
                                         },

                                         {
                                         ...
                                         },
                                         ...
                                        ]
                             }
                            ]
                }

    }

    Args:
        output_path (str): The path to save the parsed data.

    Returns:
        None
    """
    if output_path is None:
        output_path = BASE_DIR / "resources/cath_domain_boundaries.json"

    assert input_path.exists(), f"Database file not found at {input_path}"

    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    with open(input_path, "r") as f:
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
                idx += 1  # enter into the segment block

                segments = []
                for segment_idx in range(1, num_segments + 1):
                    if idx + 5 >= len(tokens):
                        break  # malformed – stop processing
                    # token pattern: chain start_i insert_start chain end_i insert_end
                    segments.append(
                        {
                            "segment_idx": segment_idx,
                            "start": int(
                                tokens[idx + 1]
                            ),  # access the start residue index in the segment block
                            "end": int(
                                tokens[idx + 4]
                            ),  # access the end residude index in the segment block
                        }
                    )
                    idx += 6  # access the next segment block

                result.setdefault(pdb_id, {}).setdefault(chain_id, []).append(
                    {
                        "domain_idx": domain_idx,
                        "segments": segments,
                    }
                )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Successfully processed CATH data and saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(process_cath_domains)
