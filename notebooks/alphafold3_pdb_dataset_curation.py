# %% [markdown]
# # Curating AlphaFold 3 PDB Dataset
#
# For training AlphaFold 3, we follow the training procedure outlined in Abramson et al (2024).
#
# Filtering of targets:
# 1. The structure must have been released to the PDB before the cutoff date of 2021-09-30.
# 2. The structure must have a reported resolution of 9 Å or less.
# 3. The maximum number of polymer chains in a considered structure is 300 for training and 1000 for evaluation.
# 4. Any polymer chain containing fewer than 4 resolved residues is filtered out.
#
# Filtering of bioassemblies:
# 1. Hydrogens are removed.
# 2. Polymer chains with all unknown residues are removed.
# 3. Clashing chains are removed. Clashing chains are defined as those with >30% of atoms within 1.7 Å of an atom
# in another chain. If two chains are clashing with each other, the chain with the greater percentage of clashing
# atoms will be removed. If the same fraction of atoms are clashing, the chain with fewer total atoms is removed.
# If the chains have the same number of atoms, then the chain with the larger chain id is removed.
# 4. For residues or small molecules with CCD codes, atoms outside of the CCD code’s defined set of atom names are
# removed.
# 5. Leaving atoms (ligand atom or groups of atoms that detach when bonds form) for covalent ligands are filtered
# out.
# 6. Protein chains with consecutive Cα atoms >10 Å apart are filtered out.
# 7. For bioassemblies with greater than 20 chains, we select a random interface token (with a centre atom <15 Å to
# the centre atom of a token in another chain) and select the closest 20 chains to this token based on minimum
# distance between any tokens centre atom.
# 8. Crystallization aids are removed if the mmCIF method information indicates that crystallography was used (see
# Table 9).
#

# %%
import argparse
import glob
import os

import pandas as pd
from Bio.PDB import PDBIO, MMCIFParser, PDBParser

# Define cutoff date
cutoff_date = pd.to_datetime("2021-09-30")


# Function to parse structures based on file type
def parse_structure(file_path):
    if file_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif file_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format")
    structure_id = file_path.split("/")[-1].split(".")[0]
    structure = parser.get_structure(structure_id, file_path)
    return structure


# Function to filter based on resolution
def filter_resolution(structure, max_resolution=9.0):
    if (
        hasattr(structure.header, "resolution")
        and structure.header["resolution"] <= max_resolution
    ):
        return True
    return False


# Function to filter based on number of polymer chains
def filter_polymer_chains(structure, max_chains=300, for_training=True):
    count = sum(1 for chain in structure.get_chains() if chain.id[0] == " ")
    return count <= (max_chains if for_training else 1000)


# Function to filter polymer chains based on resolved residues
def filter_resolved_residues(structure):
    for chain in structure.get_chains():
        if len([res for res in chain if res.id[0] == " "]) < 4:
            structure[0].detach_child(chain.id)
    return structure


# Function to remove hydrogens
def remove_hydrogens(structure):
    for chain in structure.get_chains():
        for res in chain:
            for atom in res.get_atoms():
                if atom.element == "H":
                    res.detach_child(atom.name)
    return structure


# Function to remove polymer chains with unknown residues
def remove_unknown_residues(structure):
    for chain in structure.get_chains():
        if all(res.resname == "UNK" for res in chain):
            structure[0].detach_child(chain.id)
    return structure


# Function to remove clashing chains
def remove_clashing_chains(structure):
    chains = list(structure.get_chains())
    clash_threshold = 1.7
    clash_percentage = 0.3
    clashing_chains = []

    for i, chain1 in enumerate(chains):
        for chain2 in chains[i + 1 :]:
            clash_count = sum(
                1
                for atom1 in chain1.get_atoms()
                for atom2 in chain2.get_atoms()
                if (atom1 - atom2) < clash_threshold
            )
            if (
                clash_count / len(list(chain1.get_atoms())) > clash_percentage
                or clash_count / len(list(chain2.get_atoms())) > clash_percentage
            ):
                clashing_chains.append((chain1, chain2, clash_count))

    for chain1, chain2, clash_count in clashing_chains:
        if clash_count / len(list(chain1.get_atoms())) > clash_count / len(
            list(chain2.get_atoms())
        ):
            structure[0].detach_child(chain1.id)
        elif clash_count / len(list(chain2.get_atoms())) > clash_count / len(
            list(chain1.get_atoms())
        ):
            structure[0].detach_child(chain2.id)
        else:
            if len(list(chain1.get_atoms())) > len(list(chain2.get_atoms())):
                structure[0].detach_child(chain2.id)
            else:
                structure[0].detach_child(chain1.id)
    return structure


# Function to remove atoms not in CCD code set
def remove_non_ccd_atoms(structure, ccd_atoms):
    for chain in structure.get_chains():
        for res in chain:
            for atom in res.get_atoms():
                if atom.name not in ccd_atoms.get(res.resname, []):
                    res.detach_child(atom.name)
    return structure


# Function to remove leaving atoms in covalent ligands
def remove_leaving_atoms(structure, covalent_ligands):
    for chain in structure.get_chains():
        for res in chain:
            if res.resname in covalent_ligands:
                for atom in res.get_atoms():
                    if atom.name in covalent_ligands[res.resname]:
                        res.detach_child(atom.name)
    return structure


# Function to filter chains with large Cα distances
def filter_large_ca_distances(structure):
    for chain in structure.get_chains():
        ca_atoms = [res["CA"] for res in chain if "CA" in res]
        for i, ca1 in enumerate(ca_atoms[:-1]):
            ca2 = ca_atoms[i + 1]
            if (ca1, ca2) > 10:
                structure[0].detach_child(chain.id)
                break
    return structure


# Function to select closest 20 chains in large bioassemblies
def select_closest_chains(structure, max_chains=20):
    if len(structure.get_chains()) > max_chains:
        chains = list(structure.get_chains())
        import random

        token_chain = random.choice(chains)
        token_atom = random.choice(list(token_chain.get_atoms()))
        chain_distances = []
        for chain in chains:
            min_distance = min(token_atom - atom for atom in chain.get_atoms())
            chain_distances.append((chain, min_distance))
        chain_distances.sort(key=lambda x: x[1])
        for chain, _ in chain_distances[max_chains:]:
            structure[0].detach_child(chain.id)
    return structure


# Function to remove crystallization aids
def remove_crystallization_aids(structure, crystallography_methods):
    if structure.header["method"] in crystallography_methods:
        aids = [
            res
            for res in structure.get_residues()
            if res.resname in crystallography_methods[structure.header["method"]]
        ]
        for aid in aids:
            aid.parent.detach_child(aid.id)
    return structure


# Example main function to process a list of PDB/MMCIF files
def process_structures(file_paths):
    processed_structures = []
    for file_path in file_paths:
        structure = parse_structure(file_path)
        if filter_resolution(structure):
            if filter_polymer_chains(structure):
                structure = filter_resolved_residues(structure)
                structure = remove_hydrogens(structure)
                structure = remove_unknown_residues(structure)
                structure = remove_clashing_chains(structure)
                # Assuming ccd_atoms and covalent_ligands are predefined dictionaries
                structure = remove_non_ccd_atoms(structure, ccd_atoms)
                structure = remove_leaving_atoms(structure, covalent_ligands)
                structure = filter_large_ca_distances(structure)
                structure = select_closest_chains(structure)
                structure = remove_crystallization_aids(structure, crystallography_methods)
                processed_structures.append(structure)
    return processed_structures


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Process mmCIF files to curate the AlphaFold 3 PDB dataset."
)
parser.add_argument(
    "--mmcif_dir",
    type=str,
    default=os.path.join("data", "mmCIF"),
    help="Path to the input directory containing mmCIF files to process.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.path.join("data", "PDB_set"),
    help="Path to the output directory in which to store processed mmCIF dataset files.",
)
args = parser.parse_args("")

# Example usage
file_paths = glob.glob(os.path.join(args.input_dir, "*", "*.cif"))
ccd_atoms = {"ALA": ["N", "CA", "C", "O"], "GLY": ["N", "CA", "C", "O"]}
covalent_ligands = {"LIG": ["H1", "H2"]}
crystallography_methods = {"X-RAY DIFFRACTION": ["HOH", "SO4"]}
processed_structures = process_structures(file_paths)

# Save processed structures
io = PDBIO()
os.makedirs(args.output_dir, exist_ok=True)
for structure in processed_structures:
    io.set_structure(structure)
    io.save(os.path.join(args.output_dir, f"{structure.id}.cif"))
