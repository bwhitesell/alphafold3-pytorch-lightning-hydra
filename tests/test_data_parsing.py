"""This file prepares unit tests for data parsing (e.g., mmCIF file I/O)."""

import os
import random

import pytest
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from alphafold3_pytorch.data import mmcif_parsing

os.environ["TYPECHECK"] = "True"


@pytest.mark.parametrize("mmcif_dir", [os.path.join("data", "pdb_data", "unfiltered_mmcifs")])
@pytest.mark.parametrize("complex_id", ["100d", "1k7a", "4xij", "6adq", "7a4d", "8a3j"])
def test_unfiltered_mmcif_object_parsing(mmcif_dir: str, complex_id: str) -> None:
    """Tests mmCIF file parsing and `Biomolecule` object creation for unfiltered mmCIF files.

    :param mmcif_dir: The directory containing all (unfiltered) PDB mmCIF files.
    :param complex_id: The PDB ID of the complex to be tested.
    """
    complex_filepath = os.path.join(mmcif_dir, complex_id[1:3], f"{complex_id}.cif")

    if not os.path.exists(complex_filepath):
        pytest.skip(f"File '{complex_filepath}' does not exist.")

    with open(complex_filepath, "r") as f:
        mmcif_string = f.read()

    parsing_result = mmcif_parsing.parse(
        file_id=complex_id,
        mmcif_string=mmcif_string,
        auth_chains=True,
        auth_residues=True,
    )

    if parsing_result.mmcif_object is None:
        print(f"Failed to parse file '{complex_filepath}'.")
        raise list(parsing_result.errors.values())[0]


@pytest.mark.parametrize("mmcif_dir", [os.path.join("data", "pdb_data", "mmcifs")])
@pytest.mark.parametrize("complex_id", ["100d", "1k7a", "4xij", "6adq", "7a4d", "8a3j"])
def test_filtered_mmcif_object_parsing(mmcif_dir: str, complex_id: str) -> None:
    """Tests mmCIF file parsing and `Biomolecule` object creation for filtered mmCIF files.

    :param mmcif_dir: The directory containing all (filtered) PDB mmCIF files.
    :param complex_id: The PDB ID of the complex to be tested.
    """
    complex_filepath = os.path.join(mmcif_dir, complex_id[1:3], f"{complex_id}.cif")

    if not os.path.exists(complex_filepath):
        pytest.skip(f"File '{complex_filepath}' does not exist.")

    with open(complex_filepath, "r") as f:
        mmcif_string = f.read()

    parsing_result = mmcif_parsing.parse(
        file_id=complex_id,
        mmcif_string=mmcif_string,
        auth_chains=True,
        auth_residues=True,
    )

    if parsing_result.mmcif_object is None:
        print(f"Failed to parse file '{complex_filepath}'.")
        raise list(parsing_result.errors.values())[0]


@pytest.mark.parametrize("mmcif_dir", [os.path.join("data", "pdb_data", "unfiltered_mmcifs")])
@pytest.mark.parametrize("num_random_complexes_to_parse", [500])
@pytest.mark.parametrize("random_seed", [1])
def test_unfiltered_random_mmcif_objects_parsing(
    mmcif_dir: str,
    num_random_complexes_to_parse: int,
    random_seed: int,
) -> None:
    """Tests mmCIF file parsing and `Biomolecule` object creation for a random batch
    of unfiltered mmCIF files.

    :param mmcif_dir: The directory containing all (unfiltered) PDB mmCIF files.
    :param num_random_complexes_to_parse: The number of random complexes to parse.
    :param random_seed: The random seed for reproducibility.
    """
    random.seed(random_seed)

    if not os.path.exists(mmcif_dir):
        pytest.skip(f"Directory '{mmcif_dir}' does not exist.")

    parsing_errors = []
    failed_complex_indices = []
    failed_random_complex_filepaths = []
    mmcif_subdirs = [
        os.path.join(mmcif_dir, subdir)
        for subdir in os.listdir(mmcif_dir)
        if os.path.isdir(os.path.join(mmcif_dir, subdir))
    ]
    for complex_index in range(num_random_complexes_to_parse):
        random_mmcif_subdir = random.choice(mmcif_subdirs)
        mmcif_subdir_files = [
            os.path.join(random_mmcif_subdir, mmcif_subdir_file)
            for mmcif_subdir_file in os.listdir(random_mmcif_subdir)
            if os.path.isfile(os.path.join(random_mmcif_subdir, mmcif_subdir_file))
            and mmcif_subdir_file.endswith(".cif")
        ]

        random_complex_filepath = random.choice(mmcif_subdir_files)
        complex_id = os.path.splitext(os.path.basename(random_complex_filepath))[0]

        if not os.path.exists(random_complex_filepath):
            print(f"File '{random_complex_filepath}' does not exist.")
            continue

        with open(random_complex_filepath, "r") as f:
            mmcif_string = f.read()

        parsing_result = mmcif_parsing.parse(
            file_id=complex_id,
            mmcif_string=mmcif_string,
            auth_chains=True,
            auth_residues=True,
        )

        if parsing_result.mmcif_object is None:
            parsing_errors.append(list(parsing_result.errors.values())[0])
            failed_complex_indices.append(complex_index)
            failed_random_complex_filepaths.append(random_complex_filepath)

    if parsing_result.mmcif_object is None:
        print(
            f"Failed to parse {len(parsing_errors)} files at indices {failed_complex_indices}: '{failed_random_complex_filepaths}'."
        )
        for error in parsing_errors:
            raise error


@pytest.mark.parametrize("mmcif_dir", [os.path.join("data", "pdb_data", "mmcifs")])
@pytest.mark.parametrize("num_random_complexes_to_parse", [500])
@pytest.mark.parametrize("random_seed", [1])
def test_filtered_random_mmcif_objects_parsing(
    mmcif_dir: str,
    num_random_complexes_to_parse: int,
    random_seed: int,
) -> None:
    """Tests mmCIF file parsing and `Biomolecule` object creation for a random batch
    of filtered mmCIF files.

    :param mmcif_dir: The directory containing all (filtered) PDB mmCIF files.
    :param num_random_complexes_to_parse: The number of random complexes to parse.
    :param random_seed: The random seed for reproducibility.
    """
    random.seed(random_seed)

    if not os.path.exists(mmcif_dir):
        pytest.skip(f"Directory '{mmcif_dir}' does not exist.")

    parsing_errors = []
    failed_complex_indices = []
    failed_random_complex_filepaths = []
    mmcif_subdirs = [
        os.path.join(mmcif_dir, subdir)
        for subdir in os.listdir(mmcif_dir)
        if os.path.isdir(os.path.join(mmcif_dir, subdir))
    ]
    for complex_index in range(num_random_complexes_to_parse):
        random_mmcif_subdir = random.choice(mmcif_subdirs)
        mmcif_subdir_files = [
            os.path.join(random_mmcif_subdir, mmcif_subdir_file)
            for mmcif_subdir_file in os.listdir(random_mmcif_subdir)
            if os.path.isfile(os.path.join(random_mmcif_subdir, mmcif_subdir_file))
            and mmcif_subdir_file.endswith(".cif")
        ]

        random_complex_filepath = random.choice(mmcif_subdir_files)
        complex_id = os.path.splitext(os.path.basename(random_complex_filepath))[0]

        if not os.path.exists(random_complex_filepath):
            print(f"File '{random_complex_filepath}' does not exist.")
            continue

        with open(random_complex_filepath, "r") as f:
            mmcif_string = f.read()

        parsing_result = mmcif_parsing.parse(
            file_id=complex_id,
            mmcif_string=mmcif_string,
            auth_chains=True,
            auth_residues=True,
        )

        if parsing_result.mmcif_object is None:
            parsing_errors.append(list(parsing_result.errors.values())[0])
            failed_complex_indices.append(complex_index)
            failed_random_complex_filepaths.append(random_complex_filepath)

    if parsing_result.mmcif_object is None:
        print(
            f"Failed to parse {len(parsing_errors)} files at indices {failed_complex_indices}: '{failed_random_complex_filepaths}'."
        )
        for error in parsing_errors:
            raise error
