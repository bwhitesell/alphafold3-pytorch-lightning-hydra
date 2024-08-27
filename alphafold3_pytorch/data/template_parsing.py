import os
from typing import List

import polars as pl

from alphafold3_pytorch.common.biomolecule import Biomolecule, from_mmcif_string
from alphafold3_pytorch.utils.pylogger import RankedLogger
from alphafold3_pytorch.utils.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists

logger = RankedLogger(__name__, rank_zero_only=False)


@typecheck
def parse_m8(
    m8_filepath: str, query_id: str, mmcif_dir: str, max_templates: int | None = None
) -> List[Biomolecule]:
    """Parse an M8 file and return a list of template Biomolecule objects.

    :param m8_filepath: The path to the M8 file.
    :param query_id: The ID of the query sequence.
    :param mmcif_dir: The directory containing mmCIF files.
    :param max_templates: The (optional) maximum number of templates to return.
    :return: A list of template Biomolecule objects.
    """
    # Define the column names.
    columns = [
        "ID",
        "Template ID",
        "Identity",
        "Alignment Length",
        "Mismatches",
        "Gap Openings",
        "Query Start",
        "Query End",
        "Template Start",
        "Template End",
        "E-Value",
        "Bit Score",
        "Match String",
    ]

    # Read the M8 file as a DataFrame.
    df = pl.read_csv(m8_filepath, separator="\t", has_header=False, new_columns=columns)

    # Filter the DataFrame to only include rows where
    # (1) the template ID does not contain any part of the query ID,
    # (2) the identity is less than 1.0,
    # (3) the alignment length is greater than 0, and
    # (4) the number of templates is less than the (optional) maximum number of templates.
    df = df.filter(~pl.col("Template ID").str.contains(query_id))
    df = df.filter(pl.col("Identity") < 1.0)
    df = df.filter(pl.col("Alignment Length") > 0)

    if exists(max_templates):
        df = df.head(max_templates)

    # Load each template chain as a Biomolecule object.
    template_biomols = []
    for i in range(len(df)):
        row = df[i]
        row_template_id = row["Template ID"].item()
        template_id, template_chain = row_template_id.split("_")
        template_fpath = os.path.join(mmcif_dir, template_id[1:3], f"{template_id}-assembly1.cif")
        if not os.path.exists(template_fpath):
            continue
        try:
            with open(template_fpath, "r") as f:
                template_cif = f.read()
                template_biomol = from_mmcif_string(
                    template_cif, row_template_id, chain_ids=set(template_chain)
                )
                if len(template_biomol.atom_positions):
                    template_biomols.append(template_biomol)
        except Exception as e:
            logger.warning(f"Skipping loading template {template_id} due to: {e}")

    return template_biomols
