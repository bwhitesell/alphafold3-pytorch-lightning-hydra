"""Ligand constants used in AlphaFold."""

from alphafold3_pytorch.common import amino_acid_constants, dna_constants

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.


# All ligand residues are mapped to the unknown amino acid type index.
restype_order = {}
restype_num = len(amino_acid_constants.restypes)  # := 20.


# NB: restype_3to1 serves as a placeholder for mapping all
# ligand residues to the unknown amino acid type index.
restype_3to1 = {}

# This represents the residue chemical type (i.e., `chemtype`) index of ligand residues.
chemtype_num = dna_constants.chemtype_num + 1  # := 3.
