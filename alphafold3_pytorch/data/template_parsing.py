import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from alphafold3_pytorch.common.biomolecule import (
    Biomolecule,
    _from_mmcif_object,
    get_residue_constants,
)
from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.data.data_pipeline import GAP_ID
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
    df = df.filter((pl.col("Template End") - pl.col("Template Start")) >= 9)
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


def _extract_template_features(
    template_biomol: Biomolecule,
    mapping: Mapping[int, int],
    template_sequence: str,
    query_sequence: str,
    query_chemtype: List[str],
    num_restype_classes: int = 32,
) -> Tuple[Dict[str, Any], str | None]:
    """Parse atom positions in the target structure and align with the query.

    Atoms for each residue in the template structure are indexed to coincide
    with their corresponding residue in the query sequence, according to the
    alignment mapping provided.

    Adapted from:
    https://github.com/aqlaboratory/openfold/blob/main/openfold/data/templates.py

    :param template_biomol: `Biomolecule` representing the template.
    :param mapping: Dictionary mapping indices in the query sequence to indices in
        the template sequence.
    :param template_sequence: String describing the residue sequence for the
        template.
    :param query_sequence: String describing the residue sequence for the query.
    :param query_chemtype: List of strings describing the chemical type of each
        residue in the query sequence.
    :param num_restype_classes: The total number of residue types.

    :return: A dictionary containing the extra features derived from the template
        structure.
    """
    assert len(mapping) == len(query_sequence) == len(query_chemtype), (
        f"Mapping length {len(mapping)} must match query sequence length {len(query_sequence)} "
        f"and query chemtype length {len(query_chemtype)}."
    )

    all_atom_positions = template_biomol.atom_positions
    all_atom_mask = template_biomol.atom_mask

    all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])
    all_atom_masks = np.split(all_atom_mask, all_atom_mask.shape[0])

    template_restype = []
    template_all_atom_mask = []
    template_all_atom_positions = []

    for _, chemtype in zip(query_sequence, query_chemtype):
        # Handle residues in `query_sequence` that are not in `template_sequence`.
        query_chem_residue_constants = get_residue_constants(res_chem_index=chemtype)

        template_restype.append(GAP_ID)
        template_all_atom_mask.append(
            np.zeros(query_chem_residue_constants.atom_type_num, dtype=bool)
        )
        template_all_atom_positions.append(
            np.zeros((query_chem_residue_constants.atom_type_num, 3), dtype=np.float32)
        )

    for query_index, template_index in mapping.items():
        # NOTE: Here, we assume that the query sequence's chemical types are the same as the
        # template sequence's chemical types. This is a reasonable assumption since the template
        # sequences are chemical type-specific search results for the query sequences.
        query_chem_residue_constants = get_residue_constants(
            res_chem_index=query_chemtype[query_index]
        )

        template_restype[query_index] = query_chem_residue_constants.MSA_CHAR_TO_ID.get(
            template_sequence[template_index], query_chem_residue_constants.restype_num
        )
        template_all_atom_mask[query_index] = all_atom_masks[template_index][0]
        template_all_atom_positions[query_index] = all_atom_positions[template_index][0]

    template_restype = torch.tensor(template_restype)
    template_all_atom_mask = torch.from_numpy(np.stack(template_all_atom_mask))
    template_all_atom_positions = torch.from_numpy(np.stack(template_all_atom_positions))

    return {
        "template_restype": F.one_hot(template_restype, num_classes=num_restype_classes).float(),
        # "template_pseudo_beta_mask": torch.tensor(template_pseudo_beta_mask),
        # "template_backbone_frame_mask": torch.tensor(template_backbone_frame_mask),
        # "template_distogram": torch.tensor(template_distogram),
        # "template_unit_vector": torch.tensor(template_unit_vector),
    }


class QueryToTemplateAlignError(Exception):
    """An error indicating that the query can't be aligned to the template."""
