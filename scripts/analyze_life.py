"""A simple script to analyze life's residues."""

import json
import os
from collections import defaultdict
from typing import Literal

from rdkit import Chem

from alphafold3_pytorch.common.biomolecule import get_residue_constants
from alphafold3_pytorch.data.life import DNA_NUCLEOTIDES, HUMAN_AMINO_ACIDS, RNA_NUCLEOTIDES
from alphafold3_pytorch.utils.tensor_typing import typecheck


@typecheck
def analyze_life(residue_type: Literal["peptide", "rna", "dna"], output_filepath: str):
    """Analyze life's residues."""
    if residue_type == "peptide":
        acids = HUMAN_AMINO_ACIDS
    elif residue_type == "rna":
        acids = RNA_NUCLEOTIDES
    elif residue_type == "dna":
        acids = DNA_NUCLEOTIDES

    acid_analysis_dict = {}
    for acid in acids:
        acid_smiles = acids[acid]["smile"]
        atom_compositions = defaultdict(int)
        for atom in Chem.MolFromSmiles(acid_smiles).GetAtoms():
            atom_compositions[atom.GetSymbol()] += 1
        atom_compositions = dict(sorted(atom_compositions.items()))
        num_atoms = sum(atom_compositions.values())
        acid_analysis_dict[acid] = {
            "num_atoms": num_atoms,
            "num_atoms_without_hydroxyl_oxygen": num_atoms - 1,
            "atom_compositions": atom_compositions,
        }

    with open(output_filepath, "w") as f:
        json.dump(acid_analysis_dict, f)


@typecheck
def analyze_residue_constants(
    residue_type: Literal["peptide", "rna", "dna"], output_filepath: str
):
    """Analyze residue constants."""
    rc = get_residue_constants(residue_type)

    rc_analysis_dict = {}
    for acid in rc.restype_name_to_compact_atom_names:
        acid_atoms = rc.restype_name_to_compact_atom_names[acid]
        atom_compositions = defaultdict(int)
        for atom in acid_atoms:
            if atom:
                atom_compositions[atom[0]] += 1
        atom_compositions = dict(sorted(atom_compositions.items()))
        num_atoms = sum(atom_compositions.values())
        rc_analysis_dict[acid] = {
            "num_atoms": num_atoms,
            "atom_compositions": atom_compositions,
        }

    with open(output_filepath, "w") as f:
        json.dump(rc_analysis_dict, f)


if __name__ == "__main__":
    output_dir = os.path.join("data", "analysis")
    os.makedirs(output_dir, exist_ok=True)
    # Protein residues
    analyze_life("peptide", os.path.join(output_dir, "life_protein_residue_atoms.json"))
    analyze_residue_constants("peptide", os.path.join(output_dir, "rc_protein_residue_atoms.json"))
    # RNA residues
    analyze_life("rna", os.path.join(output_dir, "life_rna_residue_atoms.json"))
    analyze_residue_constants("rna", os.path.join(output_dir, "rc_rna_residue_atoms.json"))
    # DNA residues
    analyze_life("dna", os.path.join(output_dir, "life_dna_residue_atoms.json"))
    analyze_residue_constants("dna", os.path.join(output_dir, "rc_dna_residue_atoms.json"))
