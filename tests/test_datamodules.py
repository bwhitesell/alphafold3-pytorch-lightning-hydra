"""This file prepares unit tests for datamodules."""

import pytest
import torch

from alphafold3_pytorch.data.atom_datamodule import AtomDataModule


@pytest.mark.parametrize("batch_size", [4, 8])
def test_atom_datamodule(batch_size: int) -> None:
    """Tests `AtomDataModule` to verify that the necessary attributes were created (e.g., the
    dataloader objects), and that dtypes and batch sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"
    train_val_test_split = (16, 16, 16)

    dm = AtomDataModule(
        data_dir=data_dir, train_val_test_split=train_val_test_split, batch_size=batch_size
    )
    dm.prepare_data()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = (
        len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    )  # 2 + 2 + 2 for this example
    assert num_datapoints == sum(train_val_test_split)

    batch = next(iter(dm.train_dataloader()))
    x = batch.model_forward_dict()
    assert len(x["atom_inputs"]) == batch_size
    assert x["atom_inputs"].dtype == torch.float32
