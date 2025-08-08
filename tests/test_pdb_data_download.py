import os
import pytest
import shutil
from scripts.generate_pdb_ids import main as generate_pdb_ids_main
from scripts.process_data import main as process_data_main


@pytest.mark.extended
class TestDataPipeline:
    def setup_method(self) -> None:
        """Setup method for the test class."""
        self.tests_dir = os.path.dirname(os.path.abspath(__file__))

    @pytest.fixture
    def tmp_data_dir(self) -> str:
        # Ensure tmp_data_dir exists
        tmp_data_dir = os.path.join(self.tests_dir, "tmp_data_dir")
        os.makedirs(tmp_data_dir, exist_ok=True)
        return tmp_data_dir

    def test_generate_pdb_ids(self, tmp_data_dir) -> None:
        """Test the generate_pdb_ids script."""
        generate_pdb_ids_main(
            base_data_dir=tmp_data_dir,
            resolution=9.0,
            date="2020-05-01",
            length=1000,
        )

        pdb_ids_path = os.path.join(tmp_data_dir, "pdb_ids.txt")
        with open(pdb_ids_path, "r") as f:
            pdb_ids = f.read().split(",")
        assert len(pdb_ids) == 3785
        assert os.path.exists(pdb_ids_path)

    def test_download_pdbs(self, tmp_data_dir) -> None:
        """Test the download_pdbs script."""
        download_script = os.path.join(
            self.tests_dir, "..", "scripts", "download_pdbs.sh"
        )
        # Copy example PDB IDs file to tmp data dir
        example_pdb_ids = os.path.join(self.tests_dir, "data/pdb_ids.txt")
        copied_pdb_ids = os.path.join(tmp_data_dir, "pdb_ids.txt")
        shutil.copy(example_pdb_ids, copied_pdb_ids)

        # Remove existing cif_zipped directory if it exists
        cif_zipped_dir = os.path.join(tmp_data_dir, "cif_zipped")
        if os.path.exists(cif_zipped_dir):
            shutil.rmtree(cif_zipped_dir)

        # Count number of PDB IDs in example file
        with open(copied_pdb_ids, "r") as f:
            num_ids = len(f.read().strip().split(","))

        cmd = f"bash {download_script} -o {tmp_data_dir} -n 8 -c"
        return_code = os.system(cmd)

        assert return_code == 0, "Download script failed"
        assert os.path.exists(cif_zipped_dir), "CIF zipped directory was not created"

        # Check if the number of files in the zipped directory is correct
        assert len(os.listdir(cif_zipped_dir)) == num_ids

    def test_process_data(self, tmp_data_dir) -> None:
        """Test the process_data script."""
        # Remove existing unzipped files folder if exists
        unzipped_dir = os.path.join(tmp_data_dir, "cif_unzipped")
        if os.path.exists(unzipped_dir):
            shutil.rmtree(unzipped_dir)

        src_cif_dir = os.path.join(self.tests_dir, "data", "cif_zipped")
        dst_cif_dir = os.path.join(tmp_data_dir, "cif_zipped")

        # Remove existing zipped files folder if exists
        if os.path.exists(dst_cif_dir):
            shutil.rmtree(dst_cif_dir)

        # Copy cif_zipped folder from test data to temp directory
        if os.path.exists(dst_cif_dir):
            shutil.rmtree(dst_cif_dir)
        shutil.copytree(src_cif_dir, dst_cif_dir)

        # Process the data
        process_data_main(
            base_data_dir=tmp_data_dir,
            files_per_folder=4,
            num_cpus=4,
        )

        # Check if correct number of folders were created
        folders = os.listdir(unzipped_dir)
        expected_num_folders = 2
        expected_num_files = 4
        assert (
            len(folders) == expected_num_folders
        ), f"Expected {expected_num_folders} folders, found {len(folders)}"

        # Check if each folder has correct number of files
        for folder in folders:
            folder_path = os.path.join(unzipped_dir, folder)
            num_files = len(os.listdir(folder_path))
            assert (
                num_files == expected_num_files
            ), f"Folder {folder} has {num_files} files instead of {expected_num_files}"

        # Check if features file was created
        features_path = os.path.join(tmp_data_dir, "pdb_features.json")
        failed_path = os.path.join(tmp_data_dir, "failed_files.json")

        assert os.path.exists(features_path), "Features file was not created"
        assert os.path.exists(failed_path), "Failed files list was not created"

    def teardown_method(self) -> None:
        """Teardown method for the test class."""
        tmp_data_dir = os.path.join(self.tests_dir, "tmp_data_dir")
        if os.path.exists(tmp_data_dir):
            shutil.rmtree(tmp_data_dir)