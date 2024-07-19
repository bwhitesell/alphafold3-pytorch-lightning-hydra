import os

import pytest

from alphafold3_pytorch.data.weighted_pdb_dataset import WeightedSamplerPDB


@pytest.fixture
def dataset():
    """Return a `WeightedSamplerPDB` object."""
    interface_mapping_path = os.path.join("data", "test", "interface_cluster_mapping.csv")
    chain_mapping_paths = [
        os.path.join("data", "test", "ligand_chain_cluster_mapping.csv"),
        os.path.join("data", "test", "nucleic_acid_chain_cluster_mapping.csv"),
        os.path.join("data", "test", "peptide_chain_cluster_mapping.csv"),
        os.path.join("data", "test", "protein_chain_cluster_mapping.csv"),
    ]
    return WeightedSamplerPDB(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=4,
    )


def test_sample(dataset):
    """Test the sampling method of the `WeightedSamplerPDB` class."""
    assert len(dataset.sample(4)) == 4, "The sampled batch size does not match the expected size."
