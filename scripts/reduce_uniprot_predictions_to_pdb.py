import glob
import gzip
import os
import shutil
from collections import defaultdict
from datetime import datetime

import polars as pl
from tqdm import tqdm

from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.utils.data_utils import extract_mmcif_metadata_field


def filter_pdb_files(
    input_archive_dir: str,
    input_pdb_dir: str,
    output_dir: str,
    uniprot_to_pdb_id_mapping_filepath: str,
):
    """Remove files from a given directory if they are not associated with a PDB entry, and extract
    to a given output directory all remaining archive files while grouping them their UniProt
    accession IDs.

    :param input_archive_dir: The path to the directory containing the input archive files.
    :param input_pdb_dir: The path to the directory containing the input PDB files.
    :param output_dir: The path to the directory where the filtered archive files will be saved.
    :param uniprot_to_pdb_id_mapping_filepath: The path to the file containing the mapping of
        Uniprot IDs to PDB IDs. This file is used to filter the archive files.
    """
    os.makedirs(output_dir, exist_ok=True)

    uniprot_to_pdb_id_mapping_df = pl.read_csv(
        uniprot_to_pdb_id_mapping_filepath,
        has_header=False,
        separator="\t",
        new_columns=["uniprot_accession", "database", "pdb_id"],
    )
    uniprot_to_pdb_id_mapping_df.drop_in_place("database")

    uniprot_to_pdb_id_mapping = defaultdict(set)
    for row in uniprot_to_pdb_id_mapping_df.iter_rows():
        uniprot_to_pdb_id_mapping[row[0]].add(row[1])

    archives_to_keep = defaultdict(set)
    archive_file_pattern = os.path.join(input_archive_dir, "*model_v4.cif.gz")

    for archive_file in tqdm(
        glob.glob(archive_file_pattern),
        desc="Filtering prediction files by PDB ID association",
    ):
        archive_accession_id = os.path.splitext(os.path.basename(archive_file))[0].split("-")[1]

        if archive_accession_id in uniprot_to_pdb_id_mapping:
            archives_to_keep[archive_accession_id].add(archive_file)

    for archive_accession_id in tqdm(
        archives_to_keep,
        desc="Extracting and grouping prediction files by accession ID",
    ):
        for archive in archives_to_keep[archive_accession_id]:
            output_subdir = os.path.join(output_dir, archive_accession_id)

            # Insert maximum associated PDB release date into each prediction file
            pdb_release_date = datetime(1970, 1, 1)

            for pdb_id in list(uniprot_to_pdb_id_mapping[archive_accession_id]):
                pdb_id = pdb_id.lower()
                pdb_group_code = pdb_id[1:3]

                pdb_filepath = os.path.join(
                    input_pdb_dir, pdb_group_code, f"{pdb_id}-assembly1.cif"
                )

                if os.path.exists(pdb_filepath):
                    mmcif_object = mmcif_parsing.parse_mmcif_object(
                        filepath=pdb_filepath,
                        file_id=f"{pdb_id}-assembly1.cif",
                    )
                    mmcif_release_date = extract_mmcif_metadata_field(mmcif_object, "release_date")

                    pdb_release_date = max(
                        pdb_release_date,
                        datetime.strptime(mmcif_release_date, "%Y-%m-%d"),
                    )

            if pdb_release_date == datetime(1970, 1, 1):
                print(
                    f"Could not find PDB release date for {archive_accession_id}. Skipping this prediction..."
                )
                continue

            os.makedirs(output_subdir, exist_ok=True)

            output_file = os.path.join(
                output_subdir, os.path.basename(archive).removesuffix(".gz")
            )
            with gzip.open(archive, "rb") as f_in, open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            with open(output_file, "r") as f:
                lines = f.readlines()
                new_lines = []
                for line in lines:
                    if "_pdbx_audit_revision_history.revision_date" in line:
                        new_lines.append(line)
                        new_lines.append(f'"Structure model" 1 0 1 {pdb_release_date.date()} \n')
                    else:
                        new_lines.append(line)

            with open(output_file, "w") as f:
                f.writelines(new_lines)


if __name__ == "__main__":
    input_archive_dir = os.path.join("data", "afdb_data", "unfiltered_train_mmcifs")
    input_pdb_dir = os.path.join("data", "pdb_data", "train_mmcifs")
    output_dir = os.path.join("data", "afdb_data", "train_mmcifs")
    uniprot_to_pdb_id_mapping_filepath = os.path.join(
        "data", "afdb_data", "data_caches", "uniprot_to_pdb_id_mapping.dat"
    )
    filter_pdb_files(
        input_archive_dir, input_pdb_dir, output_dir, uniprot_to_pdb_id_mapping_filepath
    )
