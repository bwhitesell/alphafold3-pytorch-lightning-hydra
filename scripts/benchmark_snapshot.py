#!/usr/bin/env python
"""
Implementation of the evaluation procedure described in sections 6.3 - 6.4 of the 
Alphafold3 supplementary information document.

https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set 
from typing import Tuple

import torch

from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import dict_to_device
from alphafold3_pytorch.models.components.inputs import PDBDataset
from alphafold3_pytorch.models.components.inputs import pdb_input_to_molecule_input
from alphafold3_pytorch.models.components.inputs import molecule_to_atom_input
from alphafold3_pytorch.models.components.inputs import pdb_dataset_to_atom_inputs
from alphafold3_pytorch.data.pdb_datamodule import AF3DataLoader
from alphafold3_pytorch.models.alphafold3_module import Sample


# Constants.
EXPECTED_DIM_ATOM_INPUT: int = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NUM_IDENTICAL_ENTITIES_FOR_EXHAUSTIVE_SEARCH: int = 8


def group_identical_entity_asym_ids(asym_ids, entity_ids) -> Set[Tuple[int]]:
    """ Collate all the asym_ids in a complex into 'identical' entity groups. """

    identical_asym_ids = {}

    for asym_id in [x.item() for x in asym_ids.unique()]:

        # Fetch the token indices that make up the asym_id/chain.
        asym_id_idxs = torch.where(asym_ids == asym_id)

        # Get the entities that compose the chain and their counts (in chain).
        entity_ids_in_asym_id, entity_id_counts_in_asym_id = torch.unique(
            entity_ids[asym_id_idxs], 
            return_counts=True
        )

        # This composition of distinct entity ids and their frequencies in the asym id
        # characterizes which chains are "identical". We can use an immutable set as
        # an order-free key to match "identical" asyms together
        chain_identifier = frozenset(
            (value.item(), count.item()) for value, count in 
            zip(entity_ids_in_asym_id, entity_id_counts_in_asym_id)
        )

        identical_asym_ids.setdefault(chain_identifier, []).append(asym_id)

    return set([tuple(x) for x in identical_asym_ids.values()])







# Analogs of methods from lightining/hydra code so eval can be run outside a lightning context.

def prepare_batch_dict(af3: Alphafold3, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the input batch dictionary for the model.

    :param batch_dict: The input batch dictionary.
    :return: The prepared batch dictionary.
    """
    if not af3.has_molecule_mod_embeds:
        batch_dict["is_molecule_mod"] = None
    return batch_dict


if __name__ == "__main__":

    # Define the command line interface.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-model-weights",
        dest="inference_model_weights",
        type=Path,
        help=(
            "File path containing alphafold3 model weights to be used in the predictions "
            "that will be evaluated."
        ),
    )
    parser.add_argument(
        "--pdb-dataset",
        dest="pdb_dataset",
        type=Path,
        help="A path to a directory containing mmcif files to evaluate",
    )
    parser.add_argument(
        "--inference-batch-size",
        dest="inference_batch_size",
        default=1,
        required=False,
        type=int,
        help="The size of batches to use for inference.",
    )
    parser.add_argument(
        "--inference-num-dataloading-workers",
        dest="inference_num_dataloading_workers",
        default=int(mp.cpu_count() * .5),
        type=int,
        help="The number of worker processes to use for dataloading.",
    )
    parser.add_argument(
        "--inference-atoms-per-window",
        dest="inference_atoms_per_window",
        type=Optional[int],
        default=None,
        required=False,
        help="The window size to use (None) if no windowing for atom pair tensor reprs.",
    )


    # Parse the user-provided args.
    args = parser.parse_args()

    # Load the model-weights into an Alphafold3 obj.
    alphafold3 = Alphafold3.init_and_load(path=args.inference_model_weights)
    # alphafold3.to(DEVICE)
    alphafold3.eval()

    # Validate the alphafold3 dimensions meet the script's expectations.
    if alphafold3.dim_atom_inputs != EXPECTED_DIM_ATOM_INPUT:
        raise ValueError(
            f"The alphafold3 model loaded from [{args.inference_model_weights}] uses a "
            "atom input shape 'dim_atom_inputs' [{alphafold3.dim_atom_inputs}] that "
            "differs from what this script expects [{EXPECTED_DIM_ATOM_INPUT}]"
        )

    # Load the pdb file dataset into a reference obj.
    pdb_dataset = PDBDataset(folder=args.pdb_dataset)
    atom_dataset = pdb_dataset_to_atom_inputs(
        pdb_dataset=pdb_dataset,
        return_atom_dataset=True
    )

    dataloader = AF3DataLoader(
        dataset=atom_dataset,
        batch_size=args.inference_batch_size,
        num_workers=args.inference_num_dataloading_workers,
        atoms_per_window=args.inference_atoms_per_window,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    
    for batched_atom_input in dataloader:

        batch_dict = prepare_batch_dict(
            af3=alphafold3,
            batch_dict=batched_atom_input.dict()
        )
        batch_dict = dict_to_device(batch_dict, device=alphafold3.device) 

        with torch.no_grad():
            # Perform inference.
            # batch_sampled_atom_pos, logits = alphafold3(
            #     **batch_dict,
            #     return_loss=False,
            #     return_confidence_head_logits=True,
            #     return_distogram_head_logits=True,
            # )

            batch_sampled_atom_pos = torch.randn([1, 3012, 3])





        # Note: A for loop here because complexs will have irregular numbers and types of 
        # enities, meaning descriptive tensors would have irregular dimensional shapes.
        for batch_idx in range(batched_atom_input.atom_inputs.size(0)):

            # Pull out tensorized mappings between atomic indices and parent entities.
            res_idx, token_idx, asym_ids, entity_ids, sym_id = (
                batched_atom_input.additional_molecule_feats[batch_idx].unbind(dim=-1)
            )

            # Group all the asym_ids that are "identical" together.
            identical_entity_asym_id_groups = group_identical_entity_asym_ids(
                asym_ids=asym_ids,
                entity_ids=entity_ids
            )

            for entity_group in identical_entity_asym_id_groups:
                if len(entity_group) <= MAX_NUM_IDENTICAL_ENTITIES_FOR_EXHAUSTIVE_SEARCH:
                    # Exhausive search.

                else:
                    # Annealing simulation.
                    pass


        # Entity resolution logic.


        # Ligand symmetry resolution logic.



        
