import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import fire
import math
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
FIRST_SEGMENT_COLUMN = 3


class ProcessingSegmentsInDomains:

    def __init__(self):
        self.all_data = self._get_data()
        self.multi_domain_proteins = self._get_multi_domain_proteins()
        self.domain_dict = self._get_domain_dict()

    def _get_data(self) -> pd.DataFrame:
        """Returns a data frame with all the data entries of the CATH database in the cath-domain-boundaries.txt file from the Orengo
        group."""

        rows = []

        with open(BASE_DIR / "resources/cath_domain_boundaries.txt", "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            row_data = {}

            row_data["pdb_chain"] = parts[0]
            row_data["domain"] = parts[1]
            row_data["fragments"] = parts[2]

            # Add variable parts dynamically
            for i, val in enumerate(parts[3:], start=3):
                row_data[f"col_{i}"] = val

            rows.append(row_data)

        return pd.DataFrame(rows)

    def _get_multi_domain_proteins(self) -> pd.DataFrame:
        """Returns a data frame with only protein chains with more than one domain, i.e rows with D02 and greater."""

        return self.all_data[~self.all_data["domain"].str.contains("D01")]

    def _get_domain_dict(self) -> dict[str, pd.DataFrame]:
        """Initialises a dictionary 'self.domain_dict' which has keys [D01, D02, ... D20] representing collections of protein chains
        in the PDB with 01, 02, ... 20 domains respectively. The values are dataframes which include the PDB IDs and domain boundary
        information for each protein chain with 1,2, ... 20 domains respectively."""

        domain_dict = {}

        for i in range(1, 21):
            domain_key = f"D{i:02}"
            domain_dict[domain_key] = self.all_data[
                self.all_data["domain"].str.contains(domain_key)
            ]

        for key in domain_dict:
            domain_dict[key] = domain_dict[key].dropna(axis=1, how="all")

        return domain_dict

    def counting_segments_in_domains(self) -> dict[int, int]:
        """Returns a dictionary containing (no. segments in a domain):(frequency)"""

        domain_lengths = (
            {}
        )  # dictionary containing (number of segments in a domain):(frequency), here the domain lengths are
        # being measured in segments

        # Iterating over each dataframe containing X domain (X=01 to 20)
        for key, df in self.domain_dict.items():
            print(f"\nProcessing: {key}")
            last_two_digits = key[-2:]  # get last two characters as string
            domains = int(last_two_digits)

            # Iterating over the rows of each dataframe
            for index, row in df.iterrows():
                segment_column = FIRST_SEGMENT_COLUMN
                domain_counter = domains
                while domain_counter > 0:
                    number_of_segments = int(row.iloc[segment_column])
                    if (
                        number_of_segments in domain_lengths
                    ):  # checking if we have an entry for this number of segments in our dictionary
                        domain_lengths[
                            number_of_segments
                        ] += 1  # adding a frequency to the key in the dictionary which tracks domains with this number of segments
                        domain_counter -= 1  # we have analysed the first domain of this row, so we will cross it off our list
                    else:
                        domain_lengths[number_of_segments] = (
                            1  # adding a frequency to the key in the dictionary which tracks domains with this number of segments
                        )
                        domain_counter -= 1  # we have analysed the first domain of this row, so we will cross it off our list
                    segment_column = segment_column + int(6 * number_of_segments) + 1

        return domain_lengths

    def counting_contiguous_domains_in_chains(self) -> dict[int, int]:
        """Returns a dictionary containing (no. of contiguous domains within a protein that only contains single
        domains in a chain that only contains contiguous domains):(frequency)"""

        columns = self.all_data.columns
        chains_with_contiguous_domains = pd.DataFrame(columns=columns)

        for key, df in tqdm(self.domain_dict.items(), desc="Domain Types", leave=True):
            print(f"\nProcessing: {key}")
            last_two_digits = key[-2:]  # get last two characters as string
            domains = int(last_two_digits)

            for index, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Processing {key}", leave=False
            ):
                segment_column = FIRST_SEGMENT_COLUMN
                domain_counter = domains
                while domain_counter > 0:
                    number_of_segments = int(row.iloc[segment_column])
                    if number_of_segments == 1:
                        segment_column = (
                            segment_column + int(6 * number_of_segments) + 1
                        )
                        domain_counter -= 1

                    else:
                        break

                    if domain_counter == 0:
                        row = row.to_frame().T  # Convert Series to one-row DataFrame
                        chains_with_contiguous_domains = pd.concat(
                            [chains_with_contiguous_domains, row], ignore_index=True
                        )

        print(chains_with_contiguous_domains)

    def checking_counting_segments_in_domains(self) -> None:
        """Returns the number of domains in the entire database by processing each line, extracting the number of domains in
        the DXX string and adding it to a counter. Also returns the total number of domains processed in the counting_segments
        method. These two numbers being equal provides strong evidence that every domain in the database has been processed when
        the counting_segments method is called."""

        total_domains = 0

        for key, df in self.domain_dict.items():
            last_two_digits = key[-2:]  # get last two characters as string
            domains = int(last_two_digits)
            total_domains += domains * (df.shape[0])

        print(
            f"The sum of the frequencies for the number of domains with n segments: {sum(self.counting_segments_in_domains().values())}"
        )
        print(
            f"The total number of domains in the CATH database: {total_domains}"
        )  # this proves that the total domains in the database equals the sum of the the values in the dictionary: Domain_lengths, so the calculations should be correct

    def plotting_frequencies_of_pseudomultimers(
        self, pseudochains="segment", show_single_pseudochain_complexes=False
    ) -> None:
        """Returns a bar chart where the x-axis is the number of pseudochains in a pseudomultimer, and the y-axis is the frequency of pseudomultimers
        with this number of pseudochains."""

        # selecting the data relating to the specified type of pseudomultimer
        if pseudochains == "segment":
            pseudomultimers = self.counting_segments_in_domains()
            title = "Breakdown of the size and frequency of pseudomultimers where: pseudomulitmer = domain, pseudochain = segment"
        elif pseudochains == "domain":
            pseudomultimers = self.counting_contiguous_domains_in_chains()
            title = "Breakdown of the size and frequency of pseudomultimers where: pseudomultimer = chain, where all domains are contiguous, pseudochain = contiguous domain"

        # getting rid of the complexes that are composed of just one pseudomultimer, if specified
        if not show_single_pseudochain_complexes:
            pseudomultimers = {k: v for k, v in pseudomultimers.items() if k != 1}

        # plotting a bar chart of the frequency vs size of the pseudomultimers that were specified
        plt.figure(figsize=(10, 6))
        plt.bar(pseudomultimers.keys(), pseudomultimers.values())
        plt.xlabel("Number of Segments in the Domain")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    def no_of_pseudomultimers(self, pseudochains="segments") -> None:
        """Returns the number of pseudomultimers of the type that is specified"""

        if pseudochains == "segments":
            pseudomultimers = self.counting_segments_in_domains()
        elif pseudochains == "domains":
            pseudomultimers = self.counting_contiguous_domains_in_chains()
            count = 0
            for key, value in pseudomultimers.items():
                if key == 1:
                    continue  # one pseudochain does not classify as a pseudomultimer
                else:
                    count += value
        print(
            f"the number of pseudomultimers, where pseudochains are {pseudochains}, is: {count}"
        )

    def no_of_collections_of_pseudochains(self, pseudochains="segments") -> None:
        """Returns the number of subsets of pseudochains that can be made from each pseudomultimer of the specified type"""

        if pseudochains == "segments":
            pseudomultimers = self.counting_segments_in_domains()
        elif pseudochains == "domains":
            pseudomultimers = self.counting_contiguous_domains_in_chains()
            count = 0
            for key, value in pseudomultimers.items():
                size = key
                temp_count = 0
                if key == 1:
                    continue
                else:
                    while size > 1:
                        temp_count += math.comb(key, size)
                        size -= 1
                    temp_count *= value
                    count += temp_count
            print(
                f"the number of collections of pseudochains ({pseudochains}) belonging to the same pseudomultimer is : {count}"
            )


if __name__ == "__main__":
    fire.Fire(ProcessingSegmentsInDomains)
