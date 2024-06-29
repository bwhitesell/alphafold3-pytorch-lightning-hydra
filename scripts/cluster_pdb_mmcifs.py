# %% [markdown]
# # Clustering AlphaFold 3 PDB Dataset
#
# For clustering AlphaFold 3's PDB dataset, we follow the clustering procedure outlined in Abramson et al (2024).
#
# In order to reduce bias in the training and evaluation sets, clustering was performed on PDB chains and interfaces, as
# follows.
# • Chain-based clustering occur at 40% sequence homology for proteins, 100% homology for nucleic acids, 100%
# homology for peptides (<10 residues) and according to CCD identity for small molecules (i.e. only identical
# molecules share a cluster).
# • Chain-based clustering of polymers with modified residues is first done by mapping the modified residues to
# a standard residue using SCOP [23, 24] convention (https://github.com/biopython/biopython/
# blob/5ee5e69e649dbe17baefe3919e56e60b54f8e08f/Bio/Data/SCOPData.py). If the mod-
# ified residue could not be found as a mapping key or was mapped to a value longer than a single character, it was
# mapped to type unknown.
# • Interface-based clustering is a join on the cluster IDs of the constituent chains, such that interfaces I and J are
# in the same interface cluster C^interface only if their constituent chain pairs {I_1,I_2},{J_1,J_2} have the same chain
# cluster pairs {C_1^chain ,C_2^chain}.

# %%

import argparse
import glob
import os
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
import pandas as pd
import rootutils
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from alphafold3_pytorch.utils import RankedLogger
from alphafold3_pytorch.utils.tensor_typing import typecheck
from scripts.filter_pdb_mmcifs import parse_mmcif_object

log = RankedLogger(__name__, rank_zero_only=True)

# Constants

CHAIN_SEQUENCES = List[Dict[str, Dict[str, str]]]
CLUSTERING_MOLECULE_TYPE = Literal["protein", "nucleic_acid", "peptide", "ligand", "unknown"]


# Helper functions


@typecheck
def convert_residue_three_to_one(
    residue: str,
) -> Tuple[str, CLUSTERING_MOLECULE_TYPE]:
    """
    Convert a three-letter amino acid, nucleotide, or CCD code to a one-letter code (if applicable).

    NOTE: All unknown residues residues (be they protein, RNA, DNA, or ligands) are converted to 'X'.
    """
    raise NotImplementedError(
        "This function needs to be reimplemented using the `Biomolecule` data structure."
    )
    if residue in PROTEIN_CODES_3TO1:
        return PROTEIN_CODES_3TO1[residue], "protein"
    elif residue in DNA_CODES_3TO1:
        return DNA_CODES_3TO1[residue], "nucleic_acid"
    elif residue in RNA_CODES_3TO1:
        return RNA_CODES_3TO1[residue], "nucleic_acid"
    elif residue in ccd_codes:
        return residue, "ligand"
    else:
        return "X", "unknown"


@typecheck
def convert_ambiguous_residue_three_to_one(
    residue: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
) -> Tuple[str, CLUSTERING_MOLECULE_TYPE]:
    """
    Convert a three-letter amino acid, nucleotide, or CCD code to a one-letter code (if applicable).

    NOTE: All unknown residues or unmappable modified residues (be they protein, RNA, DNA, or ligands) are converted to 'X'.
    """
    raise NotImplementedError(
        "This function needs to be reimplemented using the `Biomolecule` data structure."
    )
    is_modified_protein_residue = (
        molecule_type == "protein"
        and residue in SCOP_CODES_3TO1
        and len(SCOP_CODES_3TO1[residue]) == 1
    )
    is_modified_dna_residue = (
        molecule_type == "dna"
        and residue in SCOP_CODES_3TO1
        and len(SCOP_CODES_3TO1[residue]) == 1
    )
    is_modified_rna_residue = (
        molecule_type == "rna"
        and residue in SCOP_CODES_3TO1
        and len(SCOP_CODES_3TO1[residue]) == 1
    )

    # Map modified residues to their one-letter codes, if applicable
    if is_modified_protein_residue or is_modified_dna_residue or is_modified_rna_residue:
        one_letter_mapped_residue = SCOP_CODES_3TO1[residue]
        if is_modified_protein_residue:
            mapped_residue = PROTEIN_CODES_1TO3[one_letter_mapped_residue]
        elif is_modified_dna_residue:
            mapped_residue = DNA_CODES_1TO3[one_letter_mapped_residue]
        elif is_modified_rna_residue:
            mapped_residue = RNA_CODES_1TO3[one_letter_mapped_residue]
    else:
        mapped_residue = residue

    if mapped_residue in PROTEIN_CODES_3TO1:
        return PROTEIN_CODES_3TO1[mapped_residue], "protein"
    elif mapped_residue in DNA_CODES_3TO1:
        return DNA_CODES_3TO1[mapped_residue], "nucleic_acid"
    elif mapped_residue in RNA_CODES_3TO1:
        return RNA_CODES_3TO1[mapped_residue], "nucleic_acid"
    elif mapped_residue in ccd_codes:
        return mapped_residue, "ligand"
    else:
        return "X", "unknown"


@typecheck
def parse_chain_sequences_and_interfaces_from_mmcif_file(
    filepath: str, min_num_residues_for_protein_classification: int = 10
) -> Tuple[Dict[str, str], Set[str]]:
    """
    Parse an mmCIF file and return a dictionary mapping chain IDs
    to sequences for all molecule types (i.e., proteins, nucleic acids, peptides, ligands, etc)
    as well as a set of chain ID pairs denoting structural interfaces.
    """
    assert filepath.endswith(".cif"), "The input file must be an mmCIF file."
    mmcif_object = parse_mmcif_object(filepath)
    model = mmcif_object.structure

    # NOTE: After filtering, only heavy (non-hydrogen) atoms remain in the structure
    # all_atoms = [atom for atom in structure.get_atoms()]
    # neighbor_search = NeighborSearch(all_atoms)

    sequences = {}
    interface_chain_ids = set()
    for chain in model:
        num_ligands_in_chain = 0
        one_letter_seq_tokens = []
        token_molecule_types = set()

        # First find the most common molecule type in the chain
        molecule_type_counter = Counter(
            [convert_residue_three_to_one(res.resname)[-1] for res in chain]
        )
        chain_most_common_molecule_types = molecule_type_counter.most_common(2)
        chain_most_common_molecule_type = chain_most_common_molecule_types[0][0]
        if (
            chain_most_common_molecule_type == "ligand"
            and len(chain_most_common_molecule_types) > 1
        ):
            # NOTE: Ligands may be the most common molecule type in a chain, in which case
            # the second most common molecule type is required for sequence mapping
            chain_most_common_molecule_type = chain_most_common_molecule_types[1][0]

        for res in chain:
            # Then convert each residue to a one-letter code using the most common molecule type in the chain
            one_letter_residue, molecule_type = convert_ambiguous_residue_three_to_one(
                res.resname, molecule_type=chain_most_common_molecule_type
            )
            if molecule_type == "ligand":
                num_ligands_in_chain += 1
                sequences[
                    f"{chain.id}:{molecule_type}-{res.resname}-{num_ligands_in_chain}"
                ] = one_letter_residue
            else:
                assert (
                    molecule_type == chain_most_common_molecule_type
                ), f"Residue {res.resname} in chain {chain.id} has an unexpected molecule type of `{molecule_type}` (vs. the expected molecule type of `{chain_most_common_molecule_type}`)."
                one_letter_seq_tokens.append(one_letter_residue)
                token_molecule_types.add(molecule_type)

            # TODO: Efficiently compute structural interfaces by precomputing each chain's most common molecule type
            # Find all interfaces defined as pairs of chains with minimum heavy atom (i.e. non-hydrogen) separation less than 5 Å
            # for atom in res:
            #     for neighbor in neighbor_search.search(atom.coord, 5.0, "R"):
            #         neighbor_one_letter_residue, neighbor_molecule_type = convert_ambiguous_residue_three_to_one(
            #             neighbor.resname, molecule_type=chain_most_common_molecule_type
            #         )
            #         molecule_index_postfix = f"-{res.resname}-{num_ligands_in_chain}" if molecule_type == "ligand" else ""
            #         interface_chain_ids.add(f"{chain.id}:{molecule_type}{molecule_index_postfix}+{neighbor.get_parent().get_id()}:{neighbor_molecule_type}-{neighbor.resname}-{neighbor_num_ligands_in_chain}")

        assert (
            len(one_letter_seq_tokens) > 0
        ), f"No residues found in chain {chain.id} within the mmCIF file {filepath}."

        token_molecule_types = list(token_molecule_types)
        if len(token_molecule_types) > 1:
            assert (
                len(token_molecule_types) == 2
            ), f"More than two molecule types found ({token_molecule_types}) in chain {chain.id} within the mmCIF file {filepath}."
            molecule_type = [
                molecule_type
                for molecule_type in token_molecule_types
                if molecule_type != "unknown"
            ][0]
        elif len(token_molecule_types) == 1 and token_molecule_types[0] == "unknown":
            molecule_type = "protein"
        else:
            molecule_type = token_molecule_types[0]

        if (
            molecule_type == "protein"
            and len(one_letter_seq_tokens) < min_num_residues_for_protein_classification
        ):
            molecule_type = "peptide"

        one_letter_seq = "".join(one_letter_seq_tokens)
        sequences[f"{chain.id}:{molecule_type}"] = one_letter_seq

    return sequences, interface_chain_ids


@typecheck
def parse_chain_sequences_and_interfaces_from_mmcif_directory(mmcif_dir: str) -> CHAIN_SEQUENCES:
    """
    Parse all mmCIF files in a directory and return a dictionary for each complex mapping chain IDs to sequences
    as well as a set of chain ID pairs denoting structural interfaces for each complex."""
    all_chain_sequences = []
    all_interface_chain_ids = []

    mmcif_filepaths = list(glob.glob(os.path.join(mmcif_dir, "*", "*.cif")))
    for cif_filepath in tqdm(mmcif_filepaths, desc="Parsing chain sequences"):
        structure_id = os.path.splitext(os.path.basename(cif_filepath))[0]
        (
            chain_sequences,
            interface_chain_ids,
        ) = parse_chain_sequences_and_interfaces_from_mmcif_file(cif_filepath)
        all_chain_sequences.append({structure_id: chain_sequences})
        all_interface_chain_ids.append({structure_id: interface_chain_ids})

    return all_chain_sequences, all_interface_chain_ids


@typecheck
def write_sequences_to_fasta(
    all_chain_sequences: CHAIN_SEQUENCES,
    fasta_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
) -> List[str]:
    """Write sequences of a particular molecule type to a FASTA file, and return all molecule IDs."""
    assert fasta_filepath.endswith(".fasta"), "The output file must be a FASTA file."
    fasta_filepath = fasta_filepath.replace(".fasta", f"_{molecule_type}.fasta")

    molecule_ids = []
    with open(fasta_filepath, "w") as f:
        for structure_chain_sequences in tqdm(
            all_chain_sequences, desc=f"Writing {molecule_type} FASTA chain sequence file"
        ):
            for structure_id, chain_sequences in structure_chain_sequences.items():
                for chain_id, sequence in chain_sequences.items():
                    chain_id_, molecule_type_ = chain_id.split(":")
                    molecule_type_name_and_index = molecule_type_.split("-")
                    if molecule_type_name_and_index[0] == molecule_type:
                        molecule_index_postfix = (
                            f"-{molecule_type_name_and_index[1]}-{molecule_type_name_and_index[2]}"
                            if len(molecule_type_name_and_index) == 3
                            else ""
                        )
                        molecule_id = f"{structure_id}{chain_id_}:{molecule_type_name_and_index[0]}{molecule_index_postfix}"

                        f.write(f">{molecule_id}\n{sequence}\n")
                        molecule_ids.append(molecule_id)
    return molecule_ids


@typecheck
def run_clustalo(
    input_filepath: str,
    output_filepath: str,
    distmat_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
):
    """Run Clustal Omega on the input FASTA file and write the aligned FASTA sequences and corresponding distance matrix to respective output files."""
    assert input_filepath.endswith(".fasta"), "The input file must be a FASTA file."
    assert output_filepath.endswith(".fasta"), "The output file must be a FASTA file."
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."

    input_filepath = input_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    output_filepath = output_filepath.replace(".fasta", f"_{molecule_type}.fasta")
    distmat_filepath = distmat_filepath.replace(".txt", f"_{molecule_type}.txt")

    assert os.path.isfile(input_filepath), f"Input file '{input_filepath}' does not exist."

    subprocess.run(
        [
            "clustalo",
            "-i",
            input_filepath,
            "-o",
            output_filepath,
            f"--distmat-out={distmat_filepath}",
            "--percent-id",
            "--full",
            "--force",
        ]
    )


@typecheck
def cluster_ligands_by_ccd_code(input_filepath: str, distmat_filepath: str):
    """Cluster ligands based on their CCD codes and write the resulting sequence distance matrix to a file."""
    assert input_filepath.endswith(".fasta"), "The input file must be a FASTA file."
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."

    input_filepath = input_filepath.replace(".fasta", "_ligand.fasta")
    distmat_filepath = distmat_filepath.replace(".txt", "_ligand.txt")

    # Parse the ligand FASTA input file into a dictionary
    ligands = {}
    with open(input_filepath, "r") as f:
        structure_id = None
        for line in f:
            if line.startswith(">"):
                structure_id = line[1:].strip()
                ligands[structure_id] = ""
            else:
                ligands[structure_id] += line.strip()

    # Convert ligands to a list of tuples for easier indexing
    ligand_structure_ids = list(ligands.keys())
    ligand_sequences = list(ligands.values())
    n = len(ligand_structure_ids)

    # Initialize the distance matrix efficiently
    distance_matrix = np.zeros((n, n))

    # Fill the distance matrix using only the upper triangle (symmetric)
    for i in range(n):
        for j in range(i, n):
            if ligand_sequences[i] == ligand_sequences[j]:
                distance_matrix[i, j] = 100.0
                distance_matrix[j, i] = 100.0

    # Write the ligand distance matrix to a NumPy-compatible text file
    with open(distmat_filepath, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            row = [ligand_structure_ids[i]] + list(map(str, distance_matrix[i]))
            f.write(" ".join(row) + "\n")


@typecheck
def read_distance_matrix(
    distmat_filepath: str,
    molecule_type: CLUSTERING_MOLECULE_TYPE,
) -> np.ndarray:
    """Read a distance matrix from a file and return it as a NumPy array."""
    assert distmat_filepath.endswith(".txt"), "The distance matrix file must be a text file."
    distmat_filepath = distmat_filepath.replace(".txt", f"_{molecule_type}.txt")
    assert os.path.isfile(
        distmat_filepath
    ), f"Distance matrix file '{distmat_filepath}' does not exist."

    # Convert sequence matching percentages to distances through complementation
    df = pd.read_csv(distmat_filepath, sep="\s+", header=None, skiprows=1)
    matrix = 100.0 - df.values[:, 1:].astype(float)

    return matrix


@typecheck
def cluster_interfaces(
    chain_cluster_mapping: Dict[str, np.int64], interface_chain_ids: Set[str]
) -> Dict[tuple, set]:
    """Cluster interfaces based on the cluster IDs of the chains involved."""
    interface_clusters = defaultdict(set)

    interface_chain_ids = list(interface_chain_ids)
    for chain_id_pair in interface_chain_ids:
        chain_ids = chain_id_pair.split("+")
        chain_clusters = [chain_cluster_mapping[chain_id] for chain_id in chain_ids]
        if (chain_clusters[0], chain_clusters[1]) not in interface_clusters:
            interface_clusters[(chain_clusters[0], chain_clusters[1])] = f"{chain_id_pair}"

    return interface_clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster chains and interfaces within the AlphaFold 3 PDB dataset's filtered mmCIF files."
    )
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "mmcifs"),
        help="Path to the input directory containing (filtered) mmCIF files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "pdb_data", "data_caches", "clusterings"),
        help="Path to the output FASTA file.",
    )
    args = parser.parse_args()

    # Validate input arguments
    assert os.path.isdir(args.mmcif_dir), f"mmCIF directory '{args.mmcif_dir}' does not exist."
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine paths for intermediate files

    fasta_filepath = os.path.join(args.output_dir, "sequences.fasta")
    aligned_fasta_filepath = os.path.join(args.output_dir, "aligned_sequences.fasta")
    distmat_filepath = os.path.join(args.output_dir, "distmat.txt")

    # Parse all chain sequences from mmCIF files

    (
        all_chain_sequences,
        interface_chain_ids,
    ) = parse_chain_sequences_and_interfaces_from_mmcif_directory(args.mmcif_dir)

    # Align sequences separately for each molecule type and compute each respective distance matrix

    protein_molecule_ids = write_sequences_to_fasta(
        all_chain_sequences, fasta_filepath, molecule_type="protein"
    )
    nucleic_acid_molecule_ids = write_sequences_to_fasta(
        all_chain_sequences, fasta_filepath, molecule_type="nucleic_acid"
    )
    peptide_molecule_ids = write_sequences_to_fasta(
        all_chain_sequences, fasta_filepath, molecule_type="peptide"
    )
    ligand_molecule_ids = write_sequences_to_fasta(
        all_chain_sequences, fasta_filepath, molecule_type="ligand"
    )

    run_clustalo(
        fasta_filepath,
        aligned_fasta_filepath,
        distmat_filepath,
        molecule_type="protein",
    )
    run_clustalo(
        fasta_filepath,
        aligned_fasta_filepath,
        distmat_filepath,
        molecule_type="nucleic_acid",
    )
    run_clustalo(
        fasta_filepath,
        aligned_fasta_filepath,
        distmat_filepath,
        molecule_type="peptide",
    )
    cluster_ligands_by_ccd_code(
        fasta_filepath,
        distmat_filepath,
    )

    protein_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="protein")
    nucleic_acid_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="nucleic_acid")
    peptide_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="peptide")
    ligand_dist_matrix = read_distance_matrix(distmat_filepath, molecule_type="ligand")

    # Cluster residues at sequence homology levels corresponding to each molecule type

    protein_cluster_labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=40.0 + 1e-6, metric="precomputed", linkage="complete"
    ).fit_predict(protein_dist_matrix)

    nucleic_acid_cluster_labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
    ).fit_predict(nucleic_acid_dist_matrix)

    peptide_cluster_labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
    ).fit_predict(peptide_dist_matrix)

    ligand_cluster_labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1e-6, metric="precomputed", linkage="complete"
    ).fit_predict(ligand_dist_matrix)

    # Map chain sequences to cluster IDs, and save the mappings to local (CSV) storage

    protein_chain_cluster_mapping = dict(zip(protein_molecule_ids, protein_cluster_labels))
    nucleic_acid_chain_cluster_mapping = dict(
        zip(nucleic_acid_molecule_ids, nucleic_acid_cluster_labels)
    )
    peptide_chain_cluster_mapping = dict(zip(peptide_molecule_ids, peptide_cluster_labels))
    ligand_chain_cluster_mapping = dict(zip(ligand_molecule_ids, ligand_cluster_labels))

    pd.DataFrame(
        protein_chain_cluster_mapping.items(), columns=["molecule_id", "cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "protein_chain_cluster_mapping.csv"), index=False)
    pd.DataFrame(
        nucleic_acid_chain_cluster_mapping.items(), columns=["molecule_id", "cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "nucleic_acid_chain_cluster_mapping.csv"), index=False)
    pd.DataFrame(
        peptide_chain_cluster_mapping.items(), columns=["molecule_id", "cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "peptide_chain_cluster_mapping.csv"), index=False)
    pd.DataFrame(
        ligand_chain_cluster_mapping.items(), columns=["molecule_id", "cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "ligand_chain_cluster_mapping.csv"), index=False)

    # Cluster interfaces based on the cluster IDs of the chains involved, and save the interface cluster mapping to local (CSV) storage

    interface_cluster_mapping = cluster_interfaces(
        protein_chain_cluster_mapping, interface_chain_ids
    )

    pd.DataFrame(
        interface_cluster_mapping.items(), columns=["molecule_id_pair", "interface_cluster_id"]
    ).to_csv(os.path.join(args.output_dir, "interface_cluster_mapping.csv"), index=False)
