#!/usr/bin/env python
"""
Implementation of the evaluation procedure described in sections 6.3 - 6.4 of the 
Alphafold3 supplementary information document.

https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
"""

import argparse
import itertools
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
from alphafold3_pytorch.models.components.alphafold3 import ComputeModelSelectionScore
from alphafold3_pytorch.data.pdb_datamodule import AF3DataLoader
from alphafold3_pytorch.models.alphafold3_module import Sample


# Constants.
EXPECTED_DIM_ATOM_INPUT: int = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NUM_IDENTICAL_ENTITIES_FOR_EXHAUSTIVE_SEARCH: int = 8
INT_PAD_VALUE: int = -1
FLOAT_PAD_VALUE: float = 0.0

# Supporting utils.
lddt_calc_module = ComputeModelSelectionScore()


def group_identical_entity_asym_ids(
    token_idx_to_asym_ids_map,
    token_idx_to_entity_ids_map,
) -> Set[Tuple[int]]:
    """ 
    Evaluate the token to asym and token to entity mappings of a complex. Assign the 
    complex's asym ids into 'identical' entity groups. Entity groups are defined as sets 
    of asym ids that have all equal counts of tokens from the same entity ids.
    """

    identical_asym_ids = {}

    for asym_id in [x.item() for x in token_idx_to_asym_ids_map.unique()]:

        # Fetch the token indices that make up the asym_id/chain.
        asym_id_idxs = torch.where(token_idx_to_asym_ids_map == asym_id)

        # Get the entities that compose the chain and their counts (in chain).
        entity_ids_in_asym_id, entity_id_counts_in_asym_id = torch.unique(
            token_idx_to_entity_ids_map[asym_id_idxs], 
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


def exhaustive_search_for_optimal_entity_assignments(
    entity_group_asym_ids: Tuple[int],
    atom_to_asym_id_map,
    molecule_types,
    predicted_atom_pos,    
    actual_atom_pos,
):

    # Identify all permutations of the ordering of the listed asym ids.
    entity_group_asym_id_permutations = itertools.permutations(entity_group_asym_ids)

    # Iterate through each permutation of possible asym_id swaps to find the one
    # that maximizes the lddt of the complex.
    for entity_group_asym_id_permutation in entity_group_asym_id_permutations:

        permutations_predicted_atom_pos = torch.clone(predicted_atom_pos)

        # Use the indexes of the asym ids between the provided arg and each permutation
        # to define the entity swap mapping.
        for idx, swap_asym_id in enumerate(entity_group_asym_id_permutation):

            # The asym id to be swapped with.
            base_asym_id = entity_group_asym_ids[idx]

            # Skip self-swaps.
            if base_asym_id == swap_asym_id:
                continue

            # Isolate the atomic idxs to swap values for.
            original_atom_position_idxs = torch.where(atom_to_asym_id_map == base_asym_id)
            swap_atom_position_idxs = torch.where(atom_to_asym_id_map == swap_asym_id)

            # Ensure the number of atoms composing the two asym ids are identical. If 
            # they're not, something went wrong.
            if original_atom_position_idxs.size(0) != swap_atom_position_idxs.size(0):
                raise ValueError(
                    "Something went wrong, when trying to swap asym_ids to find the "
                    "closest matching assignment to the ground truth, the num atoms "
                    "beloinging to both asym ids do not match. Thus they must not be "
                    f"identical. Found shapes [{original_atom_position_idxs.size(0)}] "
                    f"and [{swap_atom_position_idxs.size(0)}]."
                )
            
            # Perform the swap on a copy of the predictions.
            permutations_predicted_atom_pos[original_atom_position_idxs] = (
                predicted_atom_pos[swap_atom_position_idxs]
            )
            permutations_predicted_atom_pos[swap_atom_position_idxs] = (
                predicted_atom_pos[original_atom_position_idxs]
            )

        # Caluclate the lddt between the positions of the actual and predicted 
        # asym id.
        lddt = lddt_calc_module.compute_chain_pair_lddt(
            # We don't need different masks for the same asym id.
            asym_mask_a=(predicted_atom_pos != FLOAT_PAD_VALUE),
            asym_mask_b=(actual_atom_pos != FLOAT_PAD_VALUE),
            true_coords=actual_atom_pos,
            pred_coords=predicted_atom_pos,
            molecule_types=molecule_types,
        )
        print(lddt)


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
        default=2,
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
    alphafold3.to(DEVICE)
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
        drop_last=False,
        int_pad_value=INT_PAD_VALUE,
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

            batch_sampled_atom_pos = torch.randn((2, 2957, 3))


        # Note: A for loop here because complexs will have irregular numbers and types of 
        # enities, meaning descriptive tensors would have irregular dimensional shapes. 
        for batch_idx in range(batched_atom_input.atom_inputs.size(0)):

            # Pull out tensorized mappings between atomic indices and parent entities.
            _, _, padded_token_idx_to_asym_ids, padded_token_idx_to_entity_id, _= (
                batched_atom_input.additional_molecule_feats[batch_idx].unbind(dim=-1)
            )

            # Remove the padding from the mappings.
            token_idx_to_asym_ids = padded_token_idx_to_asym_ids[
                torch.where(padded_token_idx_to_asym_ids != INT_PAD_VALUE)
            ]
            token_idx_to_entity_ids = padded_token_idx_to_entity_id[
                torch.where(padded_token_idx_to_entity_id != INT_PAD_VALUE)
            ]

            # The unpadded lengths of the two mappings should be equal -- because the non
            # padded tokens should be the same for the same complex.
            if token_idx_to_asym_ids.shape != token_idx_to_entity_ids.shape:
                raise ValueError(
                    "Something wen't wrong, the unpadded asym_id and entity_id maps should "
                    "be of equal shape for a single complex, but got: "
                    f"[{token_idx_to_asym_ids.shape}] and [{token_idx_to_entity_ids.shape}]."

                )

            # Group all the asym_ids that are "identical" together.
            identical_entity_asym_id_groups = group_identical_entity_asym_ids(
                token_idx_to_asym_ids=token_idx_to_asym_ids,
                token_idx_to_entity_ids=token_idx_to_entity_ids
            )

            # Construct the mapping from atom idx to asym_id.
            atom_to_asym_id_map = token_idx_to_asym_ids.repeat_interleave(
                batched_atom_input.molecule_atom_lens[batch_idx]
            )

            if atom_to_asym_id_map.size(0) != batch_sampled_atom_pos[batch_idx].size(0):
                raise ValueError(
                    "Something went wrong, the reconstructed atom idx to asym map should "
                    "have the same number of atoms as the number of atoms whose positions "
                    f"are being predicted, but got [{atom_to_asym_id_map.size(0)}] and"
                    f"[{batch_sampled_atom_pos[batch_idx].size(0)}]"
                )

            # For each group of matching asym_ids, swap their positions to find the
            # arrangement that most closely matches the obersrved arrangement.
            for entity_group_asym_ids in identical_entity_asym_id_groups:
                if len(entity_group_asym_ids) <= MAX_NUM_IDENTICAL_ENTITIES_FOR_EXHAUSTIVE_SEARCH:
                    
                    exhaustive_search_for_optimal_entity_assignments(
                        entity_group_asym_ids=entity_group_asym_ids,
                        atom_to_asym_id_map=atom_to_asym_id_map,
                        molecule_types=batch_sampled_atom_pos.is_molecule_types[batch_idx],
                        pred_pos=batch_sampled_atom_pos[batch_idx],
                        true_pos=batched_atom_input.atom_pos[batch_idx]
                    )


                else:
                    # Annealing simulation.
                    pass


        # Entity resolution logic.


        # Ligand symmetry resolution logic.



        
