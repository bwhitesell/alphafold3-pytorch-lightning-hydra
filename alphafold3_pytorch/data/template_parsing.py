import os
from datetime import datetime
from typing import List, Literal, Tuple

import polars as pl

from alphafold3_pytorch.common.biomolecule import Biomolecule, _from_mmcif_object
from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.utils.data_utils import extract_mmcif_metadata_field
from alphafold3_pytorch.utils.pylogger import RankedLogger
from alphafold3_pytorch.utils.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists

logger = RankedLogger(__name__, rank_zero_only=False)

# Constants

TEMPLATE_TYPE = Literal["protein", "dna", "rna"]


@typecheck
def parse_m8(
    m8_filepath: str,
    template_type: TEMPLATE_TYPE,
    query_id: str,
    mmcif_dir: str,
    max_templates: int | None = None,
    num_templates: int | None = None,
    template_cutoff_date: datetime | None = None,
    randomly_sample_num_templates: bool = False,
) -> List[Tuple[Biomolecule, TEMPLATE_TYPE]]:
    """Parse an M8 file and return a list of template Biomolecule objects.

    :param m8_filepath: The path to the M8 file.
    :param template_type: The type of template to parse.
    :param query_id: The ID of the query sequence.
    :param mmcif_dir: The directory containing mmCIF files.
    :param max_templates: The (optional) maximum number of templates to return.
    :param num_templates: The (optional) number of templates to return.
    :param template_cutoff_date: The (optional) cutoff date for templates.
    :param randomly_sample_num_templates: Whether to randomly sample the number of templates to
        return.
    :return: A list of template Biomolecule objects and their template types.
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
    try:
        df = pl.read_csv(m8_filepath, separator="\t", has_header=False, new_columns=columns)
    except Exception as e:
        logger.warning(f"Skipping loading M8 file {m8_filepath} due to: {e}")
        return []

    # Filter the DataFrame to only include rows where
    # (1) the template ID does not contain any part of the query ID;
    # (2) the template's identity is between 0.1 and 0.95, exclusively;
    # (3) the alignment length is greater than 0;
    # (4) the template's length is at least 10; and
    # (5) the number of templates is less than the (optional) maximum number of templates.
    df = df.filter(~pl.col("Template ID").str.contains(query_id))
    df = df.filter((pl.col("Identity") > 0.1) & (pl.col("Identity") < 0.95))
    df = df.filter(pl.col("Alignment Length") > 0)
    df = df.filter((pl.col("Template End") - pl.col("Template Start")) >= 10)
    if exists(max_templates):
        df = df.head(max_templates)

    # Select the number of templates to return.
    if len(df) and exists(num_templates) and randomly_sample_num_templates:
        df = df.sample(min(len(df), num_templates))
    elif exists(num_templates):
        df = df.head(num_templates)

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
            template_mmcif_object = mmcif_parsing.parse_mmcif_object(
                template_fpath, row_template_id
            )
            template_release_date = extract_mmcif_metadata_field(
                template_mmcif_object, "release_date"
            )
            template_biomol = _from_mmcif_object(
                template_mmcif_object, chain_ids=set(template_chain)
            )
            if not (
                exists(template_cutoff_date)
                and datetime.strptime(template_release_date, "%Y-%m-%d") <= template_cutoff_date
            ):
                continue
            elif not exists(template_cutoff_date):
                pass
            if len(template_biomol.atom_positions):
                template_biomols.append((template_biomol, template_type))
        except Exception as e:
            logger.warning(f"Skipping loading template {template_id} due to: {e}")

    return template_biomols
