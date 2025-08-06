import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import fire
from scipy.special import comb
from tqdm import tqdm
from typing import Dict
import json

from scripts.counting_pseudomultimers import ProcessingCATH

BASE_DIR = Path(__file__).parent.parent
FIRST_SEGMENT_COLUMN = 3

class ListingPseudomultimers(ProcessingCATH):
    def __init__(self, data=None):
        super().__init__(database_path=data)

    def chain_level_pseudomultimer_dict(self):
        '''
        Creates a dictionary detailing the domain boundaries for all
        chain level pseudomultimers.
            
        Args:
            None
            
        Returns:
            Dict 
        '''
        
        raw_chain_level_pseudomultimers = self.domain_dict(data= self.chains_with_contiguous_domains())

        chain_level_pseudomultimers = {}

        for no_domains, df in raw_chain_level_pseudomultimers.items():
            if int(no_domains[-2:]) == 1:
                continue
            else:
                chain_level_pseudomultimers[str(no_domains)] = {}
                for index, row in df.iterrows():
                    pdb_id = str(row["pdb_chain"])[:4]
                    chain_character = str(row["pdb_chain"])[-1]
                    if pdb_id not in chain_level_pseudomultimers[str(no_domains)]: 
                        chain_level_pseudomultimers[str(no_domains)][pdb_id] = {} # if the pdb_id is not in the dictionary, we can add it

                    chain_level_pseudomultimers[str(no_domains)][pdb_id][chain_character] = {} # if the pdb_id is not found, neither will the chain_character be
                    segment_column = FIRST_SEGMENT_COLUMN
                    domains = int(no_domains[-2:])
                    for i in range(1, domains + 1):
                        domain_key = f"domain_{i}"
                        start = int(row.iloc[segment_column + 2])
                        end = int(row.iloc[segment_column + 5])
                        chain_level_pseudomultimers[str(no_domains)][pdb_id][chain_character][domain_key] = (start, end)
                        segment_column += 7 # every domain has only one segment which means the next domain is 7 units along

        return chain_level_pseudomultimers
        
    def chain_level_pseudomultimer_json(self):
        '''
        Creating a json file with all the information about chain level pseudomultimers from CATH

        Args:
            None
        
        Returns:
            None
        '''
        with open(BASE_DIR / "resources/chain_level_pseudomultimers.json", "w") as json_file:
            json.dump(self.chain_level_pseudomultimer_dict(), json_file, indent=4)

    def contiguous_domain_length_histogram(self):
        '''
        Plotting a histogram for the length of each contiguous domain which is contained within a chain, made
        solely from contiguous domain

        Args:
            None
        
        Returns:
            None
        '''
        contiguous_domain_lengths = []

        for no_domains, pdbs in self.chain_level_pseudomultimer_dict().items():
            for pdb_id, chains in pdbs.items():
                for chain_id, domains in chains.items():
                    for domain, (start, end) in domains.items():
                        length = end - start + 1  # inclusive
                        contiguous_domain_lengths.append(length)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.hist(contiguous_domain_lengths, bins=50, edgecolor='black')
        plt.title("Histogram of Domain Lengths", fontsize=14)
        plt.xlabel("Domain Length (residues)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Set x-axis limits to zoom in (e.g., focus on lengths < 1000)
        plt.xlim(0, 1000)  # adjust if needed
        plt.savefig(BASE_DIR / "resources/domain_length_histogram.png")
        plt.show()               


if __name__ == "__main__":
    fire.Fire(ListingPseudomultimers)