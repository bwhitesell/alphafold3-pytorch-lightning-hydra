import argparse
import os
import subprocess
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import rootutils
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from sklearn.cluster import AgglomerativeClustering

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from alphafold3_pytorch.utils.typing import typecheck


@typecheck
def parse_mmcif_file(filepath: str) -> Dict[str, str]:
    """Parse an mmCIF file and return a dictionary mapping chain IDs to sequences."""
    mmcif_dict = MMCIF2Dict(filepath)
    sequences = {}

    for chain_id in mmcif_dict["_entity_poly.pdbx_strand_id"]:
        sequence = mmcif_dict["_entity_poly.pdbx_seq_one_letter_code_can"][0].replace("\n", "")
        sequences[chain_id] = sequence

    return sequences


@typecheck
def parse_mmcif_directory(mmcif_dir: str) -> Dict[str, str]:
    """Parse all mmCIF files in a directory and return a dictionary mapping chain IDs to sequences."""
    all_sequences = {}

    for root, _, files in os.walk(mmcif_dir):
        for file in files:
            if file.endswith(".cif"):
                filepath = os.path.join(root, file)
                sequences = parse_mmcif_file(filepath)
                all_sequences.update(sequences)

    return all_sequences


@typecheck
def run_clustalo(input_filepath: str, output_filepath: str):
    """Run Clustal Omega on the input file and write the aligned sequences to the output file."""
    subprocess.run(
        [
            "clustalo",
            "-i",
            input_filepath,
            "-o",
            output_filepath,
            "--distmat-out=distmat.txt",
            "--percent-id",
            "--force",
        ]
    )


@typecheck
def write_sequences_to_fasta(sequences: Dict[str, str], fasta_filepath: str):
    """Write sequences to a FASTA file."""
    with open(fasta_filepath, "w") as f:
        for chain_id, sequence in sequences.items():
            f.write(f">{chain_id}\n{sequence}\n")


@typecheck
def read_distance_matrix(distmat_filepath: str) -> np.ndarray:
    """Read a distance matrix from a file and return it as a NumPy array."""
    df = pd.read_csv(distmat_filepath, delim_whitespace=True, header=None, skiprows=1)
    matrix = df.values[:, 1:].astype(float)
    return matrix


@typecheck
def cluster_interfaces(chain_to_cluster_mapping: Dict[str, int]) -> Dict[tuple, set]:
    """Cluster interfaces based on the cluster IDs of the chains involved."""
    interface_clusters = defaultdict(set)

    for (chain1, chain2), cluster in chain_to_cluster_mapping.items():
        if (chain1, chain2) not in interface_clusters:
            interface_clusters[(cluster[chain1], cluster[chain2])].add((chain1, chain2))

    return interface_clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        default=os.path.join("data", "PDB_set"),
        help="Path to the input directory containing (filtered) mmCIF files.",
    )
    parser.add_argument(
        "--fasta_filepath",
        type=str,
        default=os.path.join("data", "sequences.fasta"),
        help="Path to the output FASTA file.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="data",
        help="Path to the temporary directory for storing intermediate files.",
    )
    args = parser.parse_args()

    # Determine paths for intermediate files
    aligned_fasta_filepath = os.path.join(
        os.path.dirname(args.fasta_filepath), "aligned_sequences.fasta"
    )
    distmat_filepath = os.path.join(args.temp_dir, "distmat.txt")

    # Parse all sequences from mmCIF files
    all_sequences = parse_mmcif_directory(args.mmcif_dir)

    # Align sequences and compute distance matrix
    write_sequences_to_fasta(all_sequences, args.fasta_filepath)
    run_clustalo(args.fasta_filepath, aligned_fasta_filepath)
    dist_matrix = read_distance_matrix(distmat_filepath)

    # Cluster e.g., proteins at 40% sequence homology
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.60, affinity="precomputed", linkage="complete"
    )
    labels = clustering.fit_predict(dist_matrix)

    # Map sequences to cluster IDs
    chain_to_cluster_mapping = {
        chain_id: cluster_id for chain_id, cluster_id in zip(all_sequences.keys(), labels)
    }

    # Cluster interfaces based on the cluster IDs of the chains involved
    interface_clusters = cluster_interfaces(chain_to_cluster_mapping)
