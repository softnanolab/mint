import pytest
import pandas as pd
from pathlib import Path
from scripts.counting_pseudomultimers import ProcessingCATH
from scripts.listing_pseudomultimers import ListingPseudomultimers
from typing import Dict

# Get the path to the test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

test_object = ListingPseudomultimers(data= TEST_DATA_DIR / "test_cath_domain_boundaries.txt")

def test_chain_level_pseudomultimers_dict():
    # Test creating a dictionary containing all the info related to chain level pseudomultimers
    dict = test_object.chain_level_pseudomultimer_dict()

    # Check not None
    assert dict is not None

    # Check type
   # assert isinstance(dict, Dict)

    # Check the correct information is stored in dict
    assert set(dict.keys()) == {"D02"}

    assert set(dict["D02"]) == {"12e8"}

    assert set(dict["D02"]["12e8"]) == {"M"}

    assert set(dict["D02"]["12e8"]["M"]) == {"domain_1", "domain_2"}

    assert set(dict["D02"]["12e8"]["M"]["domain_1"]) == {1,107}

    assert set(dict["D02"]["12e8"]["M"]["domain_2"]) == {108,211}

    