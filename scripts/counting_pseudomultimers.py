import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import fire
from scipy.special import comb
from tqdm import tqdm
from typing import Dict

BASE_DIR = Path(__file__).parent.parent
FIRST_SEGMENT_COLUMN = 3


class ProcessingCATH:
    def __init__(self, database_path=BASE_DIR / "resources/cath_domain_boundaries.txt", first_segment_column=FIRST_SEGMENT_COLUMN):
        """
        Initializes the ProcessingSegmentsInDomains object.

        Args:
            database_path (Path): Path to the root directory containing
                'resources/cath_domain_boundaries.txt'. Defaults to BASE_DIR.
            first_segment_column (str): Column name for the first domain segment.
                Used for downstream processing. Defaults to FIRST_SEGMENT_COLUMN.

        Attributes:
            all_data (pd.DataFrame): Parsed CATH data loaded from file.
        """

        self.database_path = database_path
        self.first_segment_column = first_segment_column
        self.all_data = self._get_data()

    def _get_data(self) -> pd.DataFrame:
        """
        Loads and parses CATH domain boundary data from file.

        Reads 'cath_domain_boundaries.txt' from the database path and returns
        it as a DataFrame. Used internally to populate 'self.all_data'.

        Args:
            None

        Returns:
            pd.DataFrame: Contains rows of domain boundary data, including:
                - 'pdb_chain'
                - 'domain'
                - 'fragments'
                - any additional values as 'col_X'
        """

        rows = (
            []
        )  # create an empty list which we will append dictionaries of each row to in order to create a DataFrame

        with open(self.database_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            row_data = {}  # create a dictionary to add the data for each row into

            row_data["pdb_chain"] = parts[
                0
            ]  # the first column is "pdb_chain" and the value is the first element of parts
            row_data["domain"] = parts[
                1
            ]  # the second column is "domain" and the value is the second element of parts
            row_data["fragments"] = parts[
                2
            ]  # the third column is "fragments" and the value is the third element of parts

            # The remaining elements of parts will be added under the column names of col_X for X=3 onwards
            for i, val in enumerate(parts[3:], start=3):
                row_data[f"col_{i}"] = val

            rows.append(row_data)
        all_data = pd.DataFrame(rows)

        return all_data

    def domain_dict(self) -> Dict[str, pd.DataFrame]:
        """
        Creates and returns a dictionary of domain groupings from the CATH dataset.

        Each key corresponds to a domain count label (e.g., 'D01', 'D02', ..., 'D20'),
        and the associated value is a DataFrame containing protein chains with that
        number of domains.

        Args:
            None

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are domain labels ('D01'...'D20'),
            and values are DataFrames with PDB IDs and domain boundary information
            for chains with that number of domains. Columns with all missing values are dropped.
        """

        domain_dict = {}

        for i in range(1, 21):
            domain_key = f"D{i:02}"
            domain_dict[domain_key] = self.all_data[
                self.all_data["domain"].str.contains(domain_key)
            ]

        for key in domain_dict:
            domain_dict[key] = domain_dict[key].dropna(axis=1, how="all")

        return domain_dict

    def multi_domain_proteins(self) -> pd.DataFrame:
        """
        Filters and returns protein chains with more than one domain.

        Excludes entries with domain label 'D01' and includes only those
        with 'D02' through 'D20' in the 'domain' column.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing only protein chains that
            have more than one domain.
        """

        multi_domain_proteins = self.all_data[~self.all_data["domain"].str.contains("D01")]
        return multi_domain_proteins

    def chains_with_contiguous_domains(self) -> pd.DataFrame:
        """
        Identifies and returns protein chains with only contiguous domains.

        Iterates over all domain groupings and filters chains where each domain
        consists of a single continuous segment (i.e., number_of_segments == 1 for all domains).
        Collected rows are returned as a new DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing only protein chains with strictly
            contiguous domains across all segments.
        """

        columns = self.all_data.columns
        row_dicts = []  # collect row dicts instead of concatenating DataFrames

        for key, df in tqdm(self.domain_dict().items(), desc="Domain Types", leave=True):
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
                        segment_column = segment_column + int(6 * number_of_segments) + 1
                        domain_counter -= 1

                    else:
                        break

                    if domain_counter == 0:
                        row_dicts.append(row.to_dict())  # fast row collection

        chains_with_contiguous_domains = pd.DataFrame(row_dicts, columns=columns)
        return chains_with_contiguous_domains

    def counting_chains_of_contiguous_domains(self) -> Dict[int, int]:
        """Returns a dictionary containing (no. contiguous domains within a chain):(frequency)"""

        chain_lengths = (
            {}
        )  # a dictionary which will hold: (no. contiguous domains within a chain):(frequency)
        chains = self.chains_with_contiguous_domains()

        for index, row in chains.iterrows():
            domains = int(row.loc["domain"][-2:])
            if domains in chain_lengths:  # checking if we have an entry for this chain length
                chain_lengths[domains] += 1
            else:
                chain_lengths[
                    domains
                ] = 1  # if this number of domains in a chain is not in the dictionary already, add it

        return chain_lengths

    def counting_domains_of_segments(self) -> Dict[int, int]:
        """
        Counts how domains there are that have n segments in the dataset.

        Iterates through all domain groupings (D01 ... D20), and for each domain in each
        chain, records how many segments it contains. Returns a dictionary where
        keys are the number of segments in a domain, and values are their frequency
        across the dataset.

        Args:
            None

        Returns:
            Dict[int, int]: A dictionary mapping number of segments per domain to
            the number of times that segment count appears.
        """

        domain_lengths = (
            {}
        )  # dictionary containing (number of segments in a domain):(frequency), here the domain lengths are
        # being measured in segments

        # Iterating over each dataframe containing X domain (X=01 to 20)
        for key, df in self.domain_dict().items():
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
                        domain_lengths[
                            number_of_segments
                        ] = 1  # adding a frequency to the key in the dictionary which tracks domains with this number of segments
                        domain_counter -= 1  # we have analysed the first domain of this row, so we will cross it off our list
                    segment_column = segment_column + int(6 * number_of_segments) + 1

        return domain_lengths

    def checking_counting_domains_in_chains(self) -> None:
        """Verifies that all chains with contiguous domains has been processed correctly when calculating the number of chains with n contiguous domains.

        Compares the total number of chains with contiguous domains calculated by finding the size of the self.chains_with_contiguous_domains()
        object and the sum of the frequencies in the dictionary self.counting_chains_of_contiguous_domains()

        Args:
            None

        Returns:
            None
        """

        total_chains = len(self.chains_with_contiguous_domains())

        print(
            f"The sum of the frequencies for the number of chains with n contiguous domains: {sum(self.counting_chains_of_contiguous_domains().values())}"
        )
        print(
            f"The total number of chains with contiguous domains in the CATH database: {total_chains}"
        )  # this proves that the total chains with contiguous domains in the database equals the sum of the the values in the dictionary: chain_lengths, so the calculations should be correct

    def checking_counting_segments_in_domains(self) -> None:
        """
        Verifies that all domains in the dataset have been processed correctly when calculating the number of domains with n segments.

        Compares the total number of domains (inferred from the 'DXX' labels and row counts)
        with the sum of segment frequencies produced by `counting_domains_of_segments`.

        If the two numbers match, it provides strong evidence that every domain has been
        accounted for during segment counting.

        Args:
            None

        Returns:
            None
        """

        total_domains = 0

        for key, df in self.domain_dict.items():
            last_two_digits = key[-2:]  # get last two characters as string
            domains = int(last_two_digits)
            total_domains += domains * (df.shape[0])

        print(
            f"The sum of the frequencies for the number of domains with n segments: {sum(self.counting_domains_of_segments().values())}"
        )
        print(
            f"The total number of domains in the CATH database: {total_domains}"
        )  # this proves that the total domains in the database equals the sum of the the values in the dictionary: Domain_lengths, so the calculations should be correct

    def plotting_frequencies_of_pseudomultimers(
        self, pseudochains="segment", show_single_pseudochain_complexes=False
    ) -> None:
        """
        Plots a bar chart showing the frequency distribution of pseudomultimer sizes.

        The x-axis represents the number of pseudochains per pseudomultimer, and the
        y-axis shows the frequency of such pseudomultimers in the dataset. The type
        of pseudochain can be either 'segment' or 'domain', depending on the analysis.

        Args:
            pseudochains (str): Defines what is considered a pseudochain.
                - 'segment': Each pseudomultimer is a domain and the pseudochains are the segments inside of them.
                - 'domain': Each pseudomultimer is a complete chain with only contiguous domains, pseudochains are the contiguous domains.
                Defaults to 'segment'.

            show_single_pseudochain_complexes (bool): Whether to include complexes with
                only one pseudochain in the plot (these are not actual pseudomultimers by definition). Defaults to False.

        Returns:
            None
        """

        # selecting the data relating to the specified type of pseudomultimer
        if pseudochains == "segment":
            pseudomultimers = self.counting_domains_of_segments()
            title = "Breakdown of the size and frequency of pseudomultimers where: pseudomulitmer = domain, pseudochain = segment"
        elif pseudochains == "domain":
            pseudomultimers = self.counting_chains_of_contiguous_domains()
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

    def no_pseudomultimers(self, pseudochains="segments") -> int:
        """
        Calculates the number of pseudomultimers of a specified type.

        A pseudomultimer is defined as a structure composed of two or more pseudochains.
        The method supports two modes:
            - 'segment': Each pseudomultimer is a domain and the pseudochains are the segments inside of them.
            - 'domain': Each pseudomultimer is a complete chain with only contiguous domains, pseudochains are the contiguous domains.
            Defaults to 'segment'.

        Args:
            pseudochains (str): Defines the type of pseudochains within the pseudomultimers ('segments' or 'domains').
                Defaults to 'segments'.

        Returns:
            int: The number of pseudomultimers excluding single-pseudochain structures as these do not qualify as pseudomulitmers by definiton.
        """

        if pseudochains == "segments":
            pseudomultimers = self.counting_domains_of_segments()
        elif pseudochains == "domains":
            pseudomultimers = self.counting_chains_of_contiguous_domains()
            count = 0
            for key, value in pseudomultimers.items():
                if key == 1:
                    continue  # one pseudochain does not classify as a pseudomultimer
                else:
                    count += value
        print(f"the number of pseudomultimers, where pseudochains are {pseudochains}, is: {count}")
        return count

    def no_collections_of_pseudochains(self, pseudochains="segments") -> int:
        """
        Calculates the number of pseudochain collections within each pseudomultimer.

        For each pseudomultimer of the specified type (pseudochains are either segments or domains), this method
        computes all possible subsets of pseudochains of size >1 (i.e., combinations of
        two or more pseudochains). The total number of such subsets in the database is returned.

        Args:
            pseudochains (str): Defines the type of pseudochains ('segments' or 'domains').
                Defaults to 'segments'.

        Returns:
            int: The total number of pseudochain collections (size >1) across all
            pseudomultimers in the database of the specified type.
        """

        if pseudochains == "segments":
            pseudomultimers = self.counting_domains_of_segments()
        elif pseudochains == "domains":
            pseudomultimers = self.counting_chains_of_contiguous_domains()
            count = 0
            for key, value in pseudomultimers.items():
                size = key
                temp_count = 0
                if key == 1:
                    continue
                else:
                    while size > 1:
                        temp_count += comb(key, size, exact=True)
                        size -= 1
                    temp_count *= value
                    count += temp_count
            print(
                f"the number of collections of pseudochains ({pseudochains}) belonging to the same pseudomultimer is : {count}"
            )
        return count


if __name__ == "__main__":
    fire.Fire(ProcessingCATH)
