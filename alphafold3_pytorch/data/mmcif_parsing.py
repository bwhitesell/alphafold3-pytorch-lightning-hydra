# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parses the mmCIF file format."""
import collections
import dataclasses
import functools
import io
import logging
from typing import Any, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from Bio import PDB
from Bio.Data import PDBData

from alphafold3_pytorch.data.errors import MultipleChainsError
from alphafold3_pytorch.np import residue_constants

# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]

AtomFullId = Tuple[str, int, str, Tuple[str, int, str], Tuple[str, str]]
ResidueFullId = Tuple[str, int, str, Tuple[str, int, str]]
ChainFullId = Tuple[str, int, str]


@dataclasses.dataclass(frozen=True)
class Monomer:
    """Represents a monomer in a polymer chain."""

    id: str
    num: int


@dataclasses.dataclass(frozen=True)
class ChemComp:
    """Represents a chemical composition."""

    id: str
    type: str


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=True)
class AtomSite:
    """Represents an atom site in an mmCIF file."""

    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


@dataclasses.dataclass(frozen=True)
class CovalentBond:
    """Represents a covalent bond between two atoms."""

    ptnr1_auth_seq_id: str
    ptnr1_auth_comp_id: str
    ptnr1_auth_asym_id: str
    ptnr1_label_atom_id: str
    pdbx_ptnr1_label_alt_id: str

    ptnr2_auth_seq_id: str
    ptnr2_auth_comp_id: str
    ptnr2_auth_asym_id: str
    ptnr2_label_atom_id: str
    pdbx_ptnr2_label_alt_id: str

    leaving_atom_flag: str
    conn_type_id: str


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResiduePosition:
    """Represents a residue position in a chain."""

    chain_id: str
    residue_number: int
    insertion_code: str


@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    """Represents a residue at a given position in a chain."""

    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str


@dataclasses.dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
        file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
            files being processed.
        header: Biopython header.
        structure: Biopython structure.
            chem_comp_details: Dict mapping chain_id to a list of ChemComp. E.g.
            {'A': [ChemComp, ChemComp, ...]}
        chain_to_seqres: Dict mapping chain_id to 1 letter sequence. E.g.
            {'A': 'ABCDEFG'}
        seqres_to_structure: Dict; for each chain_id contains a mapping between
            SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                            1: ResidueAtPosition,
                                                            ...}}
        covalent_bonds: List of CovalentBond.
        raw_string: The raw string used to construct the MmcifObject.
        atoms_to_remove: Optional set of atoms to remove.
        residues_to_remove: Optional set of residues to remove.
        chains_to_remove: Optional set of chains to remove.
    """

    file_id: str
    header: PdbHeader
    structure: PdbStructure
    chem_comp_details: Mapping[ChainId, Sequence[ChemComp]]
    chain_to_seqres: Mapping[ChainId, SeqRes]
    seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    covalent_bonds: Sequence[CovalentBond]
    raw_string: Any
    atoms_to_remove: Set[AtomFullId]
    residues_to_remove: Set[ResidueFullId]
    chains_to_remove: Set[ChainFullId]


@dataclasses.dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
      mmcif_object: A MmcifObject, may be None if no chain could be successfully
        parsed.
      errors: A dict mapping (file_id, chain_id) to any exception generated.
    """

    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""


def mmcif_loop_to_list(prefix: str, parsed_info: MmCIFDict) -> Sequence[Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a list.

    Reference for loop_ in mmCIF:
      http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

    :param prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
    :param parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    :return: A list of dicts; each dict represents 1 entry from an mmCIF loop.
    """
    cols = []
    data = []
    for key, value in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    assert all([len(xs) == len(data[0]) for xs in data]), (
        "mmCIF error: Not all loops are the same length: %s" % cols
    )

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(
    prefix: str,
    index: str,
    parsed_info: MmCIFDict,
) -> Mapping[str, Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a dictionary.

    :param prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
    :param index: Which item of loop data should serve as the key.
    :param parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    :return: A dict of dicts; each dict represents 1 entry from an mmCIF loop,
      indexed by the index column.
    """
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}


@functools.lru_cache(16, typed=False)
def parse(*, file_id: str, mmcif_string: str, catch_all_errors: bool = True) -> ParsingResult:
    """Entry point, parses an mmcif_string.

    :param file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
    :param mmcif_string: Contents of an mmCIF file.
    :param catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.

    :return: A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure("", handle)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the complex chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains = _get_complex_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(None, {(file_id, ""): "No complex chains found in this file."})
        seq_start_num = {
            chain_id: min([monomer.num for monomer in seq])
            for chain_id, (seq, _) in valid_chains.items()
        }

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != "1":
                # We only process the first model at the moment.
                # TODO: Expand the first bioassembly, to obtain a biologically relevant complex (AF3 Supplement, Section 2.1).
                continue

            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

            if atom.mmcif_chain_id in valid_chains:
                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ("HOH", "WAT"):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = " "
                position = ResiduePosition(
                    chain_id=atom.author_chain_id,
                    residue_number=int(atom.author_seq_num),
                    insertion_code=insertion_code,
                )
                seq_idx = int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                current = seq_to_structure_mappings.get(atom.author_chain_id, {})
                current[seq_idx] = ResidueAtPosition(
                    position=position,
                    name=atom.residue_name,
                    is_missing=False,
                    hetflag=hetflag,
                )
                seq_to_structure_mappings[atom.author_chain_id] = current

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, (seq_info, _) in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            for idx, monomer in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(
                        position=None,
                        name=monomer.id,
                        is_missing=True,
                        hetflag=" ",
                    )

        # Extract sequence and chemical component details.
        author_chain_to_sequence = {}
        chem_comp_details = {}
        for chain_id, (seq_info, chem_comp_info) in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            seq = []
            for monomer_index, monomer in enumerate(seq_info):
                if "peptide" in chem_comp_info[monomer_index].type.lower():
                    code = PDBData.protein_letters_3to1.get(monomer.id, "X")
                elif "dna" in chem_comp_info[monomer_index].type.lower():
                    code = PDBData.nucleic_letters_3to1.get(monomer.id, "X")
                elif "rna" in chem_comp_info[monomer_index].type.lower():
                    code = PDBData.nucleic_letters_3to1.get(monomer.id, "X")
                else:
                    # For residue sequences, skip ligand residues.
                    continue
                seq.append(code if len(code) == 1 else "X")
            seq = "".join(seq)
            author_chain_to_sequence[author_chain] = seq
            chem_comp_details[author_chain] = chem_comp_info

        # Identify all covalent bonds.
        covalent_bonds = _get_covalent_bond_list(parsed_info)

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chem_comp_details=chem_comp_details,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            covalent_bonds=covalent_bonds,
            raw_string=parsed_info,
            atoms_to_remove=set(),
            residues_to_remove=set(),
            chains_to_remove=set(),
        )

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)


def _get_first_model(structure: PdbStructure) -> PdbStructure:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())


_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 21


def get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    revision_dates = parsed_info["_pdbx_audit_revision_history.revision_date"]
    return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list("_exptl.", parsed_info)
    header["structure_method"] = ",".join(
        [experiment["_exptl.method"].lower() for experiment in experiments]
    )

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    if "_pdbx_audit_revision_history.revision_date" in parsed_info:
        header["release_date"] = get_release_date(parsed_info)
    else:
        logging.warning("Could not determine release_date: %s", parsed_info["_entry.id"])

    header["resolution"] = 0.00
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                header["resolution"] = float(raw_resolution)
                break
            except ValueError:
                logging.debug("Invalid resolution format: %s", parsed_info[res_key])

    return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
    """Returns list of atom sites; contains data not present in the structure."""
    return [
        AtomSite(*site)
        for site in zip(  # pylint:disable=g-complex-comprehension
            parsed_info["_atom_site.label_comp_id"],
            parsed_info["_atom_site.auth_asym_id"],
            parsed_info["_atom_site.label_asym_id"],
            parsed_info["_atom_site.auth_seq_id"],
            parsed_info["_atom_site.label_seq_id"],
            parsed_info["_atom_site.pdbx_PDB_ins_code"],
            parsed_info["_atom_site.group_PDB"],
            parsed_info["_atom_site.pdbx_PDB_model_num"],
        )
    ]


def _get_covalent_bond_list(parsed_info: MmCIFDict) -> Sequence[CovalentBond]:
    """Returns list of covalent bonds present in the structure."""
    return [
        # Collect unique (partner) atom metadata required for each covalent bond
        # per https://mmcif.wwpdb.org/docs/sw-examples/python/html/connections3.html.
        CovalentBond(*conn)
        for conn in zip(  # pylint:disable=g-complex-comprehension
            # Partner 1
            parsed_info.get("_struct_conn.ptnr1_auth_seq_id", []),
            parsed_info.get("_struct_conn.ptnr1_auth_comp_id", []),
            parsed_info.get("_struct_conn.ptnr1_auth_asym_id", []),
            parsed_info.get("_struct_conn.ptnr1_label_atom_id", []),
            parsed_info.get("_struct_conn.pdbx_ptnr1_label_alt_id", []),
            # Partner 2
            parsed_info.get("_struct_conn.ptnr2_auth_seq_id", []),
            parsed_info.get("_struct_conn.ptnr2_auth_comp_id", []),
            parsed_info.get("_struct_conn.ptnr2_auth_asym_id", []),
            parsed_info.get("_struct_conn.ptnr2_label_atom_id", []),
            parsed_info.get("_struct_conn.pdbx_ptnr2_label_alt_id", []),
            # Connection metadata
            parsed_info.get("_struct_conn.pdbx_leaving_atom_flag", []),
            parsed_info.get("_struct_conn.conn_type_id", []),
        )
        if len(conn[-1]) and conn[-1].lower() == "covale"
    ]


def _get_complex_chains(
    *, parsed_info: Mapping[str, Any]
) -> Mapping[ChainId, Tuple[Sequence[Monomer], Sequence[ChemComp]]]:
    """Extracts polymer information for complex chains.

    :param parsed_info: _mmcif_dict produced by the Biopython parser.

    :return: A dict mapping mmcif chain id to a tuple of a list of Monomers and a list of ChemComps.
    """
    # Get polymer information for each entity in the structure.
    entity_poly_seqs = mmcif_loop_to_list("_entity_poly_seq.", parsed_info)

    polymers = collections.defaultdict(list)
    for entity_poly_seq in entity_poly_seqs:
        polymers[entity_poly_seq["_entity_poly_seq.entity_id"]].append(
            Monomer(
                id=entity_poly_seq["_entity_poly_seq.mon_id"],
                num=int(entity_poly_seq["_entity_poly_seq.num"]),
            )
        )

    # Get chemical compositions. Will allow us to identify which of these polymers
    # are proteins, DNA, RNA, or ligands.
    chem_comps = mmcif_loop_to_dict("_chem_comp.", "_chem_comp.id", parsed_info)

    # Get chains information for each entity. Necessary so that we can return a
    # dict keyed on chain id rather than entity.
    struct_asyms = mmcif_loop_to_list("_struct_asym.", parsed_info)

    entity_to_mmcif_chains = collections.defaultdict(list)
    for struct_asym in struct_asyms:
        chain_id = struct_asym["_struct_asym.id"]
        entity_id = struct_asym["_struct_asym.entity_id"]
        entity_to_mmcif_chains[entity_id].append(chain_id)

    # Identify and return all complex chains.
    valid_chains = {}
    for entity_id, seq_info in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]
        for chain_id in chain_ids:
            chem_comp_info = [
                ChemComp(
                    id=chem_comps[monomer.id]["_chem_comp.id"],
                    type=chem_comps[monomer.id]["_chem_comp.type"],
                )
                for monomer in seq_info
            ]
            valid_chains[chain_id] = (seq_info, chem_comp_info)
    return valid_chains


def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in (".", "?")


def get_atom_coords(
    mmcif_object: MmcifObject, chain_id: str, _zero_center_positions: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets atom positions and mask from a list of Biopython Residues."""
    # Locate the right chain
    chains = list(mmcif_object.structure.get_chains())
    relevant_chains = [c for c in chains if c.id == chain_id]
    if len(relevant_chains) != 1:
        raise MultipleChainsError(f"Expected exactly one chain in structure with id {chain_id}.")
    chain = relevant_chains[0]

    # Extract the coordinates
    num_res = len(mmcif_object.chain_to_seqres[chain_id])
    all_atom_positions = np.zeros([num_res, residue_constants.atom_type_num, 3], dtype=np.float32)
    all_atom_mask = np.zeros([num_res, residue_constants.atom_type_num], dtype=np.float32)
    for res_index in range(num_res):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        res_at_position = mmcif_object.seqres_to_structure[chain_id][res_index]
        if not res_at_position.is_missing:
            res = chain[
                (
                    res_at_position.hetflag,
                    res_at_position.position.residue_number,
                    res_at_position.position.insertion_code,
                )
            ]
            # TODO: Pick the largest-occupancy atom/residue for each ambiguous atom/residue.
            for atom in res.get_atoms():
                atom_name = atom.get_name()
                x, y, z = atom.get_coord()
                if atom_name in residue_constants.atom_order.keys():
                    pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                    mask[residue_constants.atom_order[atom_name]] = 1.0
                elif atom_name.upper() == "SE" and res.get_resname() == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    pos[residue_constants.atom_order["SD"]] = [x, y, z]
                    mask[residue_constants.atom_order["SD"]] = 1.0

            # Fix naming errors in arginine residues where NH2 is incorrectly
            # assigned to be closer to CD than NH1
            cd = residue_constants.atom_order["CD"]
            nh1 = residue_constants.atom_order["NH1"]
            nh2 = residue_constants.atom_order["NH2"]
            if (
                res.get_resname() == "ARG"
                and all(mask[atom_index] for atom_index in (cd, nh1, nh2))
                and (np.linalg.norm(pos[nh1] - pos[cd]) > np.linalg.norm(pos[nh2] - pos[cd]))
            ):
                pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
                mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()

        all_atom_positions[res_index] = pos
        all_atom_mask[res_index] = mask

    # TODO: Expand the first bioassembly, to obtain a biologically relevant complex (AF3 Supplement, Section 2.1).
    # mmcif_object.structure = _expand_first_model(mmcif_object.structure)

    if _zero_center_positions:
        binary_mask = all_atom_mask.astype(bool)
        translation_vec = all_atom_positions[binary_mask].mean(axis=0)
        all_atom_positions[binary_mask] -= translation_vec

    return all_atom_positions, all_atom_mask
