from functools import partial
from random import random, randrange
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from alphafold3_pytorch.models.components.attention import (
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed,
)
from alphafold3_pytorch.models.components.inputs import (
    AtomInput,
    BatchedAtomInput,
    maybe_transform_to_atom_inputs,
)
from alphafold3_pytorch.utils.model_utils import pad_at_dim
from alphafold3_pytorch.utils.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists

# dataloader and collation fn


@typecheck
def collate_inputs_to_batched_atom_input(
    inputs: List,
    int_pad_value=-1,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
) -> BatchedAtomInput:
    """
    Collate function for a list of AtomInput objects.

    :param inputs: A list of AtomInput objects.
    :param int_pad_value: The padding value for integer tensors.
    :param atoms_per_window: The number of atoms per window.
    :param map_input_fn: A function to apply to each input before collation.
    :return: A collated BatchedAtomInput object.
    """
    if exists(map_input_fn):
        inputs = [map_input_fn(i) for i in inputs]

    # go through all the inputs
    # and for any that is not AtomInput, try to transform it with the registered input type to corresponding registered function

    atom_inputs = maybe_transform_to_atom_inputs(inputs)

    # take care of windowing the atompair_inputs and atompair_ids if they are not windowed already

    if exists(atoms_per_window):
        for atom_input in atom_inputs:
            atompair_inputs = atom_input.atompair_inputs
            atompair_ids = atom_input.atompair_ids

            atompair_inputs_is_windowed = atompair_inputs.ndim == 4

            if not atompair_inputs_is_windowed:
                atom_input.atompair_inputs = full_pairwise_repr_to_windowed(
                    atompair_inputs, window_size=atoms_per_window
                )

            if exists(atompair_ids):
                atompair_ids_is_windowed = atompair_ids.ndim == 3

                if not atompair_ids_is_windowed:
                    atom_input.atompair_ids = full_attn_bias_to_windowed(
                        atompair_ids, window_size=atoms_per_window
                    )

    # separate input dictionary into keys and values

    keys = atom_inputs[0].dict().keys()
    atom_inputs = [i.dict().values() for i in atom_inputs]

    outputs = []

    for grouped in zip(*atom_inputs):
        # if all None, just return None

        not_none_grouped = [*filter(exists, grouped)]

        if len(not_none_grouped) == 0:
            outputs.append(None)
            continue

        # default to empty tensor for any Nones

        one_tensor = not_none_grouped[0]

        dtype = one_tensor.dtype
        ndim = one_tensor.ndim

        # use -1 for padding int values, for assuming int are labels - if not, handle within alphafold3

        if dtype in (torch.int, torch.long):
            pad_value = int_pad_value
        elif dtype == torch.bool:
            pad_value = False
        else:
            pad_value = 0.0

        # get the max lengths across all dimensions

        shapes_as_tensor = torch.stack(
            [Tensor(tuple(g.shape) if exists(g) else ((0,) * ndim)).int() for g in grouped], dim=-1
        )

        max_lengths = shapes_as_tensor.amax(dim=-1)

        default_tensor = torch.full(max_lengths.tolist(), pad_value, dtype=dtype)

        # pad across all dimensions

        padded_inputs = []

        for inp in grouped:
            if not exists(inp):
                padded_inputs.append(default_tensor)
                continue

            for dim, max_length in enumerate(max_lengths.tolist()):
                inp = pad_at_dim(inp, (0, max_length - inp.shape[dim]), value=pad_value, dim=dim)

            padded_inputs.append(inp)

        # stack

        stacked = torch.stack(padded_inputs)

        outputs.append(stacked)

    # reconstitute dictionary

    batched_atom_inputs = BatchedAtomInput(**dict(tuple(zip(keys, outputs))))
    return batched_atom_inputs


@typecheck
def AF3DataLoader(
    *args, atoms_per_window: int | None = None, map_input_fn: Callable | None = None, **kwargs
):
    """
    Create a `torch.utils.data.DataLoader` with the
    `collate_inputs_to_batched_atom_input` or `map_input_fn` function
    for data collation.

    :param args: The arguments to pass to `torch.utils.data.DataLoader`.
    :param atoms_per_window: The number of atoms per window.
    :param map_input_fn: A function to apply to each input before collation.
    :param kwargs: The keyword arguments to pass to `torch.utils.data.DataLoader`.
    :return: A `torch.utils.data.DataLoader` with a custom AF3 collate function.
    """
    collate_fn = partial(collate_inputs_to_batched_atom_input, atoms_per_window=atoms_per_window)

    if exists(map_input_fn):
        collate_fn = partial(collate_fn, map_input_fn=map_input_fn)

    return DataLoader(*args, collate_fn=collate_fn, **kwargs)


class MockAtomDataset(Dataset):
    """
    A dummy dataset for atomic data.

    :param num_examples: The number of examples in the dataset.
    :param max_seq_len: The maximum sequence length.
    :param atoms_per_window: The number of atoms per window.
    """

    def __init__(
        self,
        num_examples,
        max_seq_len=16,
        atoms_per_window=4,
    ):
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        self.atoms_per_window = atoms_per_window

    def __len__(self):
        """Return the length of the dataset."""
        return self.num_examples

    def __getitem__(self, idx) -> AtomInput:
        """
        Return a random `AtomInput` sample from the dataset.

        :param idx: The index of the sample.
        :return: A random `AtomInput` sample from the dataset.
        """
        seq_len = randrange(1, self.max_seq_len)
        atom_seq_len = self.atoms_per_window * seq_len

        atom_inputs = torch.randn(atom_seq_len, 77)
        atompair_inputs = torch.randn(atom_seq_len, atom_seq_len, 5)
        molecule_atom_lens = torch.randint(1, self.atoms_per_window, (seq_len,))
        additional_molecule_feats = torch.randint(0, 2, (seq_len, 5))
        additional_token_feats = torch.randn(seq_len, 2)
        is_molecule_types = torch.randint(0, 2, (seq_len, 4)).bool()
        molecule_ids = torch.randint(0, 32, (seq_len,))
        token_bonds = torch.randint(0, 2, (seq_len, seq_len)).bool()

        templates = torch.randn(2, seq_len, seq_len, 44)
        template_mask = torch.ones((2,)).bool()

        msa = torch.randn(7, seq_len, 64)
        msa_mask = None
        if random() > 0.5:
            msa_mask = torch.ones((7,)).bool()

        # required for training, but omitted on inference

        atom_pos = torch.randn(atom_seq_len, 3)
        molecule_atom_indices = molecule_atom_lens - 1

        distance_labels = torch.randint(0, 37, (seq_len, seq_len))
        pae_labels = torch.randint(0, 64, (seq_len, seq_len))
        pde_labels = torch.randint(0, 64, (seq_len, seq_len))
        plddt_labels = torch.randint(0, 50, (seq_len,))
        resolved_labels = torch.randint(0, 2, (seq_len,))

        return AtomInput(
            atom_inputs=atom_inputs,
            atompair_inputs=atompair_inputs,
            molecule_ids=molecule_ids,
            token_bonds=token_bonds,
            molecule_atom_lens=molecule_atom_lens,
            additional_molecule_feats=additional_molecule_feats,
            additional_token_feats=additional_token_feats,
            is_molecule_types=is_molecule_types,
            templates=templates,
            template_mask=template_mask,
            msa=msa,
            msa_mask=msa_mask,
            atom_pos=atom_pos,
            molecule_atom_indices=molecule_atom_indices,
            distance_labels=distance_labels,
            pae_labels=pae_labels,
            pde_labels=pde_labels,
            plddt_labels=plddt_labels,
            resolved_labels=resolved_labels,
        )


class AtomDataModule(LightningDataModule):
    """`LightningDataModule` for a dummy atomic dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (2, 2, 2),
        sequence_crop_size: int = 384,
        sampling_weight_for_disorder_pdb_distillation: float = 0.02,
        train_on_transcription_factor_distillation_sets: bool = False,
        pdb_distillation: Optional[bool] = None,
        max_number_of_chains: int = 20,
        atoms_per_window: int | None = None,
        map_dataset_input_fn: Optional[Callable] = None,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        # if map dataset function given, curry into DataLoader

        self.dataloader_class = partial(AF3DataLoader, atoms_per_window=atoms_per_window)

        if exists(map_dataset_input_fn):
            self.dataloader_class = partial(
                self.dataloader_class, map_input_fn=map_dataset_input_fn
            )

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load dataset splits only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = MockAtomDataset(num_examples=self.hparams.train_val_test_split[0])
            self.data_val = MockAtomDataset(num_examples=self.hparams.train_val_test_split[1])
            self.data_test = MockAtomDataset(num_examples=self.hparams.train_val_test_split[2])

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.dataloader_class(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self.dataloader_class(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self.dataloader_class(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = AtomDataModule()
