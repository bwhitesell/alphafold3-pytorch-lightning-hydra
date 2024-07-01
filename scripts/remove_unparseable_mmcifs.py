import argparse
import os

from tqdm import tqdm

from alphafold3_pytorch.common.biomolecule import _from_mmcif_object
from alphafold3_pytorch.data import mmcif_parsing


def remove_unparseable_mmcifs(mmcif_dir: str, dry_run: bool = False):
    """
    Delete unparseable mmCIF files from a directory.

    :param mmcif_dir: directory containing mmCIF files
    :param dry_run: if True, print the unparseable mmCIF files without deleting them
    """
    assert os.path.exists(mmcif_dir), f"Directory '{mmcif_dir}' does not exist."

    parsing_errors = []
    unparseable_complex_filepaths = []
    mmcif_subdirs = [
        os.path.join(mmcif_dir, subdir)
        for subdir in os.listdir(mmcif_dir)
        if os.path.isdir(os.path.join(mmcif_dir, subdir))
        and os.listdir(os.path.join(mmcif_dir, subdir))
    ]
    for mmcif_subdir in tqdm(mmcif_subdirs, desc="Parsing mmCIF subdirectories"):
        mmcif_subdir_files = [
            os.path.join(mmcif_subdir, mmcif_subdir_file)
            for mmcif_subdir_file in os.listdir(mmcif_subdir)
            if os.path.isfile(os.path.join(mmcif_subdir, mmcif_subdir_file))
            and mmcif_subdir_file.endswith(".cif")
        ]

        for complex_filepath in mmcif_subdir_files:
            complex_id = os.path.splitext(os.path.basename(complex_filepath))[0]

            with open(complex_filepath, "r") as f:
                mmcif_string = f.read()

            parsing_result = mmcif_parsing.parse(
                file_id=complex_id,
                mmcif_string=mmcif_string,
                auth_chains=True,
                auth_residues=True,
            )

            if parsing_result.mmcif_object is None:
                parsing_errors.append(list(parsing_result.errors.values())[0])
                unparseable_complex_filepaths.append(complex_filepath)
                if not dry_run:
                    os.remove(complex_filepath)
            else:
                try:
                    biomol = _from_mmcif_object(parsing_result.mmcif_object)
                except Exception as e:
                    if "mmCIF contains an insertion code" in str(e):
                        continue
                    else:
                        parsing_errors.append(e)
                        unparseable_complex_filepaths.append(complex_filepath)
                        if not dry_run:
                            os.remove(complex_filepath)
                        continue
                if len(biomol.atom_positions) == 0:
                    parsing_errors.append(
                        AssertionError(
                            f"Failed to parse file '{complex_filepath}' into a `Biomolecule` object."
                        )
                    )
                    unparseable_complex_filepaths.append(complex_filepath)
                    if not dry_run:
                        os.remove(complex_filepath)

    if parsing_errors:
        print(f"Failed to parse {len(parsing_errors)} files: '{unparseable_complex_filepaths}'.")
        for error in parsing_errors:
            print(error)
        raise error


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "mmcif_dir",
        type=str,
        help="Directory containing mmCIF files to validate.",
    )
    args.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the unparseable mmCIF files without deleting them.",
    )
    args = args.parse_args()

    remove_unparseable_mmcifs(args.mmcif_dir, args.dry_run)
