import pytest
import pandas as pd
from pathlib import Path
from scripts.counting_pseudomultimers import ProcessingCATH

# Get the path to the test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

test_object = ProcessingCATH(database_path= TEST_DATA_DIR / "test_cath_domain_boundaries.txt" )

def test_get_data():
    # Test creating a DataFrame from cath_domain_boundaries.txt 
    all_data = test_object._get_data()

    # Check not None
    assert all_data is not None

    # Check type
    assert isinstance(all_data, pd.DataFrame)

    # Check number of rows 
    assert all_data.shape[0] == 6

    # Check expected column names
    expected_base_columns = ["pdb_chain", "domain", "fragments"]
    for col in expected_base_columns:
        assert col in all_data.columns

    expected_cols = [f"col_{i}" for i in range(3, 37)]
    for col in expected_cols:
        assert col in all_data.columns, f"Missing column: {col}"

def test_domain_dict():
    # Test creation of dictionary of a dictionary containing keys: ["D01" ... "D20"] and values: DataFrames containing chains 
    domain_groups = test_object.domain_dict()

    # Check not None
    assert domain_groups is not None

    # Check that we have keys from D01 to D20
    expected_keys = [f"D{i:02}" for i in range(1, 21)]
    assert list(domain_groups.keys()) == expected_keys

    # Checking each value of each key
    for key, df in domain_groups.items():

        # Check that each value is a DataFrame
        assert isinstance(df, pd.DataFrame)

        # Some of the value DataFrames will be empty
        if not df.empty:
            # Check that every row's 'domain' column contains the domain key
            assert df["domain"].str.contains(key).all(), f"Unexpected domain in {key}"
        
        if key == "D01":
            # Check that there are 2 rows in the DataFrame for chains with 1 domain
            assert df.shape[0] == 2
        
        if key == "D02":
            # Check that there are 4 rows in the DataFrame for chains with 2 domain 
            assert df.shape[0] == 4
        

def test_chains_with_contiguous_domains():
    # Checking the identification of chains with only contiguous domains
    chains_with_contiguous_domains = test_object.chains_with_contiguous_domains()

    # Check not None
    assert chains_with_contiguous_domains is not None

    # Check the type 
    assert isinstance(chains_with_contiguous_domains, pd.DataFrame)

    # Define the expected set of PDB chains
    expected_pdb_chains = {"101mA", "102lA", "12e8M"}

    # Get the actual set of PDB chains from the result
    actual_pdb_chains = set(chains_with_contiguous_domains["pdb_chain"])

    # Assert exact match (no more, no less)
    assert actual_pdb_chains == expected_pdb_chains, (
        f"Expected PDB chains {expected_pdb_chains}, but got {actual_pdb_chains}"
    )

