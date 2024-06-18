"""Ligand constants used in AlphaFold."""

from typing import Final

from alphafold3_pytorch.common import amino_acid_constants, dna_constants

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    # NOTE: Taken from: https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L117C13-L126C18
    "Al",
    "As",
    "Au",
    "B",
    "Be",
    "Br",
    "C",
    "Ca",
    "Cl",
    "Co",
    "Cr",
    "Cu",
    "F",
    "Fe",
    "Hg",
    "I",
    "Ir",
    "K",
    "Li",
    "Mg",
    "Mn",
    "Mo",
    "N",
    "Ni",
    "O",
    "Os",
    "P",
    "Pb",
    "Pd",
    "Pr",
    "Pt",
    "Re",
    "Rh",
    "Ru",
    "S",
    "Sb",
    "Se",
    "Si",
    "Sn",
    "Tb",
    "Te",
    "U",
    "W",
    "V",
    "Y",
    "Zn",
    "ATM",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 47.


# All ligand residues are mapped to the unknown amino acid type index (:= 20).
restypes = ["UNK"]
restype_order = {
    restype: len(amino_acid_constants.restypes) + i for i, restype in enumerate(restypes)
}
restype_num = len(amino_acid_constants.restypes)  # := 20.

BIOMOLECULE_CHAIN: Final[str] = "other"
POLYMER_CHAIN: Final[str] = "non-polymer"


# NB: restype_3to1 serves as a placeholder for mapping all
# ligand residues to the unknown amino acid type index (:= 20).
restype_3to1 = {}

# Define a restype name for all unknown ligand residues.
unk_restype = "UNK"

# This represents the residue chemical type (i.e., `chemtype`) index of ligand residues.
chemtype_num = dna_constants.chemtype_num + 1  # := 3.