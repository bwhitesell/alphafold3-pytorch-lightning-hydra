"""This file prepares unit tests for data parsing (e.g., mmCIF file I/O)."""

import os
import random

import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from alphafold3_pytorch.data import mmcif_parsing

os.environ["TYPECHECK"] = "True"


@pytest.mark.parametrize("mmcif_dir", [os.path.join("data", "pdb_data", "mmcifs")])
@pytest.mark.parametrize("complex_id", ["6adq", "100d", "1k7a"])
@pytest.mark.parametrize("auth_chains", [True])
@pytest.mark.parametrize("auth_residues", [True])
def test_mmcif_object_parsing(
    mmcif_dir: str, complex_id: str, auth_chains: bool, auth_residues: bool
) -> None:
    """Tests mmCIF file parsing and `Biomolecule` object creation.

    :param mmcif_dir: The directory containing all (filtered) PDB mmCIF files.
    :param complex_id: The PDB ID of the complex to be tested.
    :param auth_chains: Whether to use the 'auth_asym_id' field as chain ID.
    :param auth_residues: Whether to use the 'auth_seq_id' field as residue ID.
    """
    complex_filepath = os.path.join(mmcif_dir, complex_id[1:3], f"{complex_id}.cif")

    if os.path.exists(complex_filepath):
        with open(complex_filepath, "r") as f:
            mmcif_string = f.read()

        parsing_result = mmcif_parsing.parse(
            file_id=complex_id,
            mmcif_string=mmcif_string,
            auth_chains=auth_chains,
            auth_residues=auth_residues,
        )

        if parsing_result.mmcif_object is None:
            print(f"Failed to parse file '{complex_filepath}'.")
            raise list(parsing_result.errors.values())[0]
    else:
        pytest.skip(f"File '{complex_filepath}' does not exist.")


@pytest.mark.parametrize("mmcif_dir", [os.path.join("data", "pdb_data", "mmcifs")])
@pytest.mark.parametrize("num_random_complexes_to_parse", [25])
@pytest.mark.parametrize("random_seed", [1])
@pytest.mark.parametrize("auth_chains", [True])
@pytest.mark.parametrize("auth_residues", [True])
def test_mmcif_objects_parsing(
    mmcif_dir: str,
    num_random_complexes_to_parse: int,
    random_seed: int,
    auth_chains: bool,
    auth_residues: bool,
) -> None:
    """Tests mmCIF file parsing and `Biomolecule` object creation with (random) batch parsing.

    :param mmcif_dir: The directory containing all (filtered) PDB mmCIF files.
    :param num_random_complexes_to_parse: The number of random complexes to parse.
    :param random_seed: The random seed for reproducibility.
    :param auth_chains: Whether to use the 'auth_asym_id' field as chain ID.
    :param auth_residues: Whether to use the 'auth_seq_id' field as residue ID.
    """
    random.seed(random_seed)

    mmcif_subdirs = [
        os.path.join(mmcif_dir, subdir)
        for subdir in os.listdir(mmcif_dir)
        if os.path.isdir(os.path.join(mmcif_dir, subdir))
    ]
    for _ in range(num_random_complexes_to_parse):
        random_mmcif_subdir = random.choice(mmcif_subdirs)
        mmcif_subdir_files = [
            os.path.join(random_mmcif_subdir, mmcif_subdir_file)
            for mmcif_subdir_file in os.listdir(random_mmcif_subdir)
            if os.path.isfile(os.path.join(random_mmcif_subdir, mmcif_subdir_file))
            and mmcif_subdir_file.endswith(".cif")
        ]

        random_complex_filepath = random.choice(mmcif_subdir_files)
        complex_id = os.path.splitext(os.path.basename(random_complex_filepath))[0]

        if os.path.exists(random_complex_filepath):
            with open(random_complex_filepath, "r") as f:
                mmcif_string = f.read()

            parsing_result = mmcif_parsing.parse(
                file_id=complex_id,
                mmcif_string=mmcif_string,
                auth_chains=auth_chains,
                auth_residues=auth_residues,
            )

            if parsing_result.mmcif_object is None:
                print(f"Failed to parse file '{random_complex_filepath}'.")
                raise list(parsing_result.errors.values())[0]
        else:
            pytest.skip(f"File '{random_complex_filepath}' does not exist.")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
