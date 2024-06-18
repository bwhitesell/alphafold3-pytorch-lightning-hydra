import collections
import dataclasses
import functools
import io
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Model import Model

from alphafold3_pytorch.common import (
    amino_acid_constants,
    dna_constants,
    ligand_constants,
    mmcif_metadata,
    rna_constants,
)
from alphafold3_pytorch.data import mmcif_parsing
from alphafold3_pytorch.utils.typing import IntType, typecheck
from alphafold3_pytorch.utils.utils import exists, np_mode

MMCIF_PREFIXES_TO_DROP_POST_PARSING = [
    "_atom_site.",
    "_chem_comp.",
    "_entity.",
    "_entity_poly.",
    "_struct_asym.",
]
MMCIF_PREFIXES_TO_DROP_POST_AF3 = MMCIF_PREFIXES_TO_DROP_POST_PARSING + [
    "_citation.",
    "_citation_author.",
]


@dataclasses.dataclass(frozen=True)
class Biomolecule:
    """Biomolecule structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid or nucleotide type for each residue represented as an integer
    # between 0 and 31, where:
    # 20 represents the the unknown amino acid 'X';
    # 25 represents the unknown RNA nucleotide `N`;
    # 30 represents the unknown DNA nucleotide `DN`;
    # and 31 represents the gap token `-`.
    restype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the biomolecule that this
    # residue belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of the atoms of each residue
    # (in sq. angstroms units), representing the displacement of the
    # residue's atoms from their ground truth mean values.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chemical ID of each amino-acid, nucleotide, or ligand residue represented
    # as a string. This is primarily used to record a ligand residue's name
    # (e.g., when exporting an mmCIF file from a Biomolecule object).
    chemid: np.ndarray  # [num_res]

    # Chemical type of each amino-acid, nucleotide, or ligand residue represented
    # as an integer between 0 and 3. This is used to determine whether a residue is
    # a protein (0), RNA (1), DNA (2), or ligand (3) residue.
    chemtype: np.ndarray  # [num_res]

    # Chemical component details of each residue as a unique `ChemComp` object.
    # This is used to determine the biomolecule's unique chemical IDs, types, and names.
    # N.b., this is primarily used to record chemical component metadata
    # (e.g., when exporting an mmCIF file from a Biomolecule object).
    chem_comp_table: Set[mmcif_parsing.ChemComp]  # [num_unique_chem_comp]

    # Raw mmCIF metadata dictionary parsed for the biomolecule using Biopython.
    # N.b., this is primarily used to retain mmCIF assembly metadata
    # (e.g., when exporting an mmCIF file from a Biomolecule object).
    mmcif_metadata: Dict[str, Any]  # [1]


@typecheck
def get_residue_constants(
    res_chem_type: Optional[str] = None, res_chem_index: Optional[IntType] = None
) -> ModuleType:
    """Returns the corresponding residue constants for a given residue chemical type."""
    assert exists(res_chem_type) or exists(
        res_chem_index
    ), "Either `res_chem_type` or `res_chem_index` must be provided."
    if (exists(res_chem_type) and "peptide" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 0
    ):
        residue_constants = amino_acid_constants
    elif (exists(res_chem_type) and "rna" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 1
    ):
        residue_constants = rna_constants
    elif (exists(res_chem_type) and "dna" in res_chem_type.lower()) or (
        exists(res_chem_index) and res_chem_index == 2
    ):
        residue_constants = dna_constants
    else:
        residue_constants = ligand_constants
    return residue_constants


@typecheck
def _from_mmcif_object(
    mmcif_object: mmcif_parsing.MmcifObject, chain_id: Optional[str] = None
) -> Biomolecule:
    """Takes a Biopython structure/model mmCIF object and creates a `Biomolecule` instance.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    :param mmcif_object: The parsed Biopython structure/model mmCIF object.
    :param chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
        Otherwise all chains are parsed.

    :return: A new `Biomolecule` created from the structure/model mmCIF object contents.

    :raise:
      ValueError: If the number of models included in a given structure is not 1.
      ValueError: If insertion code is detected at a residue.
    """
    structure = mmcif_object.structure
    if isinstance(structure, Model):
        model = structure
    else:
        models = list(structure.get_models())
        if len(models) != 1:
            raise ValueError(
                "Only single model mmCIFs are supported. Found" f" {len(models)} models."
            )
        model = models[0]

    atom_positions = []
    restype = []
    chemid = []
    chemtype = []
    residue_chem_comp_details = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res_index, res in enumerate(chain):
            if res.id[2] != " ":
                raise ValueError(
                    f"mmCIF contains an insertion code at chain {chain.id} and"
                    f" residue index {res.id[1]}. These are not supported."
                )
            res_chem_comp_details = mmcif_object.chem_comp_details[chain.id][res_index]
            residue_constants = get_residue_constants(res_chem_type=res_chem_comp_details.type)
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            restype.append(restype_idx)
            chemid.append(res_chem_comp_details.id)
            chemtype.append(residue_constants.chemtype_num)
            residue_chem_comp_details.append(res_chem_comp_details)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Biomolecule(
        atom_positions=np.array(atom_positions),
        restype=np.array(restype),
        atom_mask=np.array(atom_mask),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        chemid=np.array(chemid),
        chemtype=np.array(chemtype),
        chem_comp_table=set(residue_chem_comp_details),
        mmcif_metadata=mmcif_object.raw_string,
    )


@typecheck
def from_mmcif_string(mmcif_str: str, file_id: str, chain_id: Optional[str] = None) -> Biomolecule:
    """Takes a mmCIF string and constructs a `Biomolecule` object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    :param mmcif_str: The contents of the mmCIF file.
    :param file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
    :param chain_id: If chain_id is specified (e.g. A), then only that chain is parsed.
        Otherwise all chains are parsed.

    :return: A new `Biomolecule` parsed from the mmCIF contents.

    :raise:
        ValueError: If the mmCIF file is not valid.
    """
    mmcif_object = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_str)

    # Crash if an error is encountered. Any parsing errors should have
    # been dealt with beforehand (e.g., at the alignment stage).
    if mmcif_object.mmcif_object is None:
        raise list(mmcif_object.errors.values())[0]

    return _from_mmcif_object(mmcif_object.structure, mmcif_object.mmcif_object, chain_id)


@typecheck
def atom_id_to_type(atom_id: str) -> str:
    """Convert atom ID to atom type, works only for standard residues.

    :param atom_id: Atom ID to be converted.
    :return: String corresponding to atom type.

    :raise:
        ValueError: If `atom_id` is empty or does not contain any alphabetic characters.
    """
    assert len(atom_id) > 0, f"Atom ID must have at least one character, but received: {atom_id}."
    for char in atom_id:
        if char.isalpha():
            return char
    raise ValueError(
        f"Atom ID must contain at least one alphabetic character, but received: {atom_id}."
    )


@typecheck
def remove_metadata_fields_by_prefixes(
    metadata_dict: Dict[str, List[Any]], field_prefixes: List[str]
) -> Dict[str, List[Any]]:
    """Remove metadata fields from the metadata dictionary.

    :param metadata_dict: The metadata default dictionary from which to remove metadata fields.
    :param field_prefixes: A list of prefixes to remove from the metadata default dictionary.

    :return: A metadata dictionary with the specified metadata fields removed.
    """
    return {
        key: value
        for key, value in metadata_dict.items()
        if not any(key.startswith(prefix) for prefix in field_prefixes)
    }


@typecheck
def to_mmcif(
    biomol: Biomolecule, file_id: str, insert_alphafold_mmcif_metadata: bool = True
) -> str:
    """Converts a `Biomolecule` instance to an mmCIF string.

    WARNING 1: The _entity_poly_seq is filled with unknown residues for any
      missing residue indices in the range from min(1, min(residue_index)) to
      max(residue_index).
      E.g. for a biomolecule object with positions for (majority) protein residues
      2 (MET), 3 (LYS), 6 (GLY), this method would set the _entity_poly_seq to:
      1 UNK
      2 MET
      3 LYS
      4 UNK
      5 UNK
      6 GLY
      This is done to preserve the residue numbering.

    WARNING 2: Converting ground truth mmCIF file to Biomolecule and then back to
      mmCIF using this method will convert all non-standard residue types to UNK.
      If you need this behaviour, you need to store more mmCIF metadata in the
      Biomolecule object (e.g. all fields except for the _atom_site loop).

    WARNING 3: Converting ground truth mmCIF file to Biomolecule and then back to
      mmCIF using this method will not retain the original chain indices.

    WARNING 4: In case of multiple identical chains, they are assigned different
      `_atom_site.label_entity_id` values.

    :param biomol: A biomolecule to convert to mmCIF string.
    :param file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
    :param insert_alphafold_mmcif_metadata: If True, insert metadata fields
        referencing AlphaFold in the output mmCIF file.

    :return: A valid mmCIF string.

    :raise:
      ValueError: If amino-acid or nucleotide types array contains entries with
      too many biomolecule types.
    """
    atom_positions = biomol.atom_positions
    restype = biomol.restype
    atom_mask = biomol.atom_mask
    residue_index = biomol.residue_index.astype(np.int32)
    chain_index = biomol.chain_index.astype(np.int32)
    b_factors = biomol.b_factors
    chemid = biomol.chemid
    chemtype = biomol.chemtype

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    # We count unknown residues as protein residues.
    for entity_id in np.unique(chain_index):  # np.unique gives sorted output.
        chain_ids[entity_id] = _int_id_to_str_id(entity_id + 1)

    mmcif_dict = collections.defaultdict(list)

    mmcif_dict["data_"] = file_id.upper()
    mmcif_dict["_entry.id"] = file_id.upper()

    label_asym_id_to_entity_id = {}
    # Entity and chain information.
    for entity_id, chain_id in chain_ids.items():
        # Determine the chemical type of the chain.
        res_chem_index = np_mode(chemtype[chain_index == entity_id])[0].item()
        residue_constants = get_residue_constants(res_chem_index=res_chem_index)
        # Add all chain information to the _struct_asym table.
        label_asym_id_to_entity_id[str(chain_id)] = str(entity_id)
        mmcif_dict["_struct_asym.id"].append(chain_id)
        mmcif_dict["_struct_asym.entity_id"].append(str(entity_id))
        # Add information about the entity to the _entity_poly table.
        mmcif_dict["_entity_poly.entity_id"].append(str(entity_id))
        mmcif_dict["_entity_poly.type"].append(residue_constants.BIOMOLECULE_CHAIN)
        mmcif_dict["_entity_poly.pdbx_strand_id"].append(chain_id)
        # Generate the _entity table.
        mmcif_dict["_entity.id"].append(str(entity_id))
        mmcif_dict["_entity.type"].append(residue_constants.POLYMER_CHAIN)

    # Add the polymer residues to the _entity_poly_seq table.
    for entity_id, (res_ids, aa_chemids) in _get_entity_poly_seq(
        restype,
        residue_index,
        chain_index,
        chemid,
        chemtype,
    ).items():
        for res_id, aa_chemid in zip(res_ids, aa_chemids):
            mmcif_dict["_entity_poly_seq.entity_id"].append(str(entity_id))
            mmcif_dict["_entity_poly_seq.hetero"].append("n")
            mmcif_dict["_entity_poly_seq.num"].append(str(res_id))
            mmcif_dict["_entity_poly_seq.mon_id"].append(aa_chemid)

    # Populate the chem comp table.
    for chem_comp in biomol.chem_comp_table:
        mmcif_dict["_chem_comp.id"].append(chem_comp.id)
        mmcif_dict["_chem_comp.type"].append(chem_comp.type)
        mmcif_dict["_chem_comp.name"].append(chem_comp.name)

    # Add all atom sites.
    atom_index = 1
    for i in range(restype.shape[0]):
        # Determine the chemical type of the residue.
        residue_constants = get_residue_constants(res_chem_index=chemtype[i])
        res_name_3 = chemid[i]
        if (restype[i] - residue_constants.min_restype_num) <= len(residue_constants.restypes):
            atom_names = residue_constants.atom_types
        else:
            raise ValueError(
                "Amino acid types array contains entries with too many protein types."
            )
        for atom_name, pos, mask, b_factor in zip(
            atom_names, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue
            type_symbol = atom_id_to_type(atom_name)

            mmcif_dict["_atom_site.group_PDB"].append("ATOM")
            mmcif_dict["_atom_site.id"].append(str(atom_index))
            mmcif_dict["_atom_site.type_symbol"].append(type_symbol)
            mmcif_dict["_atom_site.label_atom_id"].append(atom_name)
            mmcif_dict["_atom_site.label_alt_id"].append(".")
            mmcif_dict["_atom_site.label_comp_id"].append(res_name_3)
            mmcif_dict["_atom_site.label_asym_id"].append(chain_ids[chain_index[i]])
            mmcif_dict["_atom_site.label_entity_id"].append(
                label_asym_id_to_entity_id[chain_ids[chain_index[i]]]
            )
            mmcif_dict["_atom_site.label_seq_id"].append(str(residue_index[i]))
            mmcif_dict["_atom_site.pdbx_PDB_ins_code"].append(".")
            mmcif_dict["_atom_site.Cartn_x"].append(f"{pos[0]:.3f}")
            mmcif_dict["_atom_site.Cartn_y"].append(f"{pos[1]:.3f}")
            mmcif_dict["_atom_site.Cartn_z"].append(f"{pos[2]:.3f}")
            mmcif_dict["_atom_site.occupancy"].append("1.00")
            mmcif_dict["_atom_site.B_iso_or_equiv"].append(f"{b_factor:.2f}")
            mmcif_dict["_atom_site.auth_seq_id"].append(str(residue_index[i]))
            mmcif_dict["_atom_site.auth_asym_id"].append(chain_ids[chain_index[i]])
            mmcif_dict["_atom_site.pdbx_PDB_model_num"].append("1")

            atom_index += 1

    init_metadata_dict = (
        remove_metadata_fields_by_prefixes(biomol.mmcif_metadata, MMCIF_PREFIXES_TO_DROP_POST_AF3)
        if insert_alphafold_mmcif_metadata
        else remove_metadata_fields_by_prefixes(
            biomol.mmcif_metadata, MMCIF_PREFIXES_TO_DROP_POST_PARSING
        )
    )
    metadata_dict = mmcif_metadata.add_metadata_to_mmcif(
        mmcif_dict, insert_alphafold_mmcif_metadata=insert_alphafold_mmcif_metadata
    )
    init_metadata_dict.update(metadata_dict)
    init_metadata_dict.update(mmcif_dict)

    return _create_mmcif_string(init_metadata_dict)


@typecheck
@functools.lru_cache(maxsize=256)
def _int_id_to_str_id(num: IntType) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    :param num: A positive integer.

    :return: A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(output)


@typecheck
def _get_entity_poly_seq(
    restypes: np.ndarray,
    residue_indices: np.ndarray,
    chain_indices: np.ndarray,
    chemids: np.ndarray,
    chemtypes: np.ndarray,
) -> Dict[IntType, Tuple[List[IntType], List[str]]]:
    """Constructs gapless residue index and chemid lists for each chain.

    :param restypes: A numpy array with restypes.
    :param residue_indices: A numpy array with residue indices.
    :param chain_indices: A numpy array with chain indices.
    :param chemids: A numpy array with residue chemical IDs.
    :param chemtypes: A numpy array with residue chemical types.

    :return: A dictionary mapping chain indices to a tuple with a list of residue indices
      and a list of chemids. Missing residues are filled with the unknown residue type
      of the corresponding residue chemical type (e.g., `N`, the unknown RNA type,
      for a majority-RNA chain). Ligand residues are not present in the output.
    """
    if (
        restypes.shape[0] != residue_indices.shape[0]
        or restypes.shape[0] != chain_indices.shape[0]
    ):
        raise ValueError("restypes, residue_indices, chain_indices must have the same length.")

    # Group the present residues by chain index.
    present = collections.defaultdict(list)
    num_ligand_residues_present = collections.defaultdict(int)
    for chain_id, res_id, aa, aa_chemtype in zip(
        chain_indices, residue_indices, restypes, chemtypes
    ):
        if aa_chemtype == 3:
            # Skip ligand residues.
            num_ligand_residues_present[chain_id] += 1
            continue
        present[chain_id].append((res_id - num_ligand_residues_present[chain_id], aa))

    # Add any missing residues (from 1 to the first residue and for any gaps).
    entity_poly_seq = {}
    for chain_id, present_residues in present.items():
        present_residue_indices = set([x[0] for x in present_residues])
        min_res_id = min(present_residue_indices)  # Could be negative.
        max_res_id = max(present_residue_indices)

        res_chem_index = np_mode(chemtypes[chain_indices == chain_id])[0].item()
        residue_constants = get_residue_constants(res_chem_index=res_chem_index)

        new_residue_indices = []
        new_chemids = []
        present_index = 0
        for i in range(min(1, min_res_id), max_res_id + 1):
            new_residue_indices.append(i)
            if i in present_residue_indices:
                new_chemids.append(chemids[chain_indices == chain_id][present_index])
                present_index += 1
            else:
                # Unknown residue type of the most common residue chemical type in the chain.
                new_chemids.append(residue_constants.unk_restype)
        entity_poly_seq[chain_id] = (new_residue_indices, new_chemids)
    return entity_poly_seq


@typecheck
def _create_mmcif_string(mmcif_dict: Dict[str, Any]) -> str:
    """Converts mmCIF dictionary into mmCIF string."""
    mmcifio = MMCIFIO()
    mmcifio.set_dict(mmcif_dict)

    with io.StringIO() as file_handle:
        mmcifio.save(file_handle)
        return file_handle.getvalue()
