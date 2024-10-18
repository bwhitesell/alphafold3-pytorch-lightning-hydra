#!/usr/bin/env python
"""
Implementation of the evaluation procedure described in sections 6.3 - 6.4 of the 
Alphafold3 supplementary information document.

https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
"""

import argparse
import enum
import itertools
import math
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set 
from typing import Tuple
from typing import TypedDict

from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import dict_to_device
from alphafold3_pytorch.models.components.inputs import PDBDataset
from alphafold3_pytorch.models.components.inputs import pdb_dataset_to_atom_inputs
from alphafold3_pytorch.models.components.inputs import IS_PROTEIN
from alphafold3_pytorch.models.components.inputs import IS_DNA
from alphafold3_pytorch.models.components.inputs import IS_RNA
from alphafold3_pytorch.models.components.inputs import IS_LIGAND
from alphafold3_pytorch.models.components.inputs import IS_RNA_INDEX 
from alphafold3_pytorch.models.components.inputs import IS_DNA_INDEX
from alphafold3_pytorch.models.components.inputs import IS_LIGAND_INDEX
from alphafold3_pytorch.models.components.inputs import IS_MOLECULE_TYPES
from alphafold3_pytorch.models.components.alphafold3 import ComputeModelSelectionScore
from alphafold3_pytorch.models.components.alphafold3 import WeightedRigidAlign 
from alphafold3_pytorch.data.pdb_datamodule import AF3DataLoader
from alphafold3_pytorch.utils.tensor_typing import Bool
from alphafold3_pytorch.utils.tensor_typing import Float
from alphafold3_pytorch.utils.tensor_typing import Int
import torch


# Constants.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NUM_IDENTICAL_ENTITIES_FOR_EXHAUSTIVE_SEARCH: int = 3
EXPECTED_DIM_ATOM_INPUT: int = 3
INT_PAD_VALUE: int = -1
FLOAT_PAD_VALUE: float = 0.0


class PolymerType(str, enum.Enum):
    PROTEIN: str = "PROTEIN"
    NUCLEIC_ACID: str = "NUCLEIC_ACID"
    LIGAND: str = "LIGAND"

    @classmethod
    def from_molecular_idx(cls, idx: int) -> "PolymerType":
        if idx == IS_PROTEIN:
            return cls.PROTEIN
        elif idx == IS_RNA:
            return cls.NUCLEIC_ACID
        elif idx == IS_DNA:
            return cls.NUCLEIC_ACID
        elif idx == IS_LIGAND:
            return cls.LIGAND
        else:
            raise ValueError(
                f"Tried to resolve a polymer type from an unknown molecular idx [{idx}]"
            )


class ComplexType(str, enum.Enum):
    PROTEIN_PROTIEN: str = "PROTEIN_PROTEIN"
    NUCLEICACID_NUCLEICACID: str = "NUCLEICACID_NUCLEICACID"
    NUCLEICACID_PROTEIN: str = "NUCLEICACID_PROTEIN"
    LIGAND_PROTEIN: str = "LIGAND_PROTEIN"
    LIGAND_NUCLEICACID: str = "LIGAND_NUCLEICACID"

    def is_eligible_polymer(self, p: PolymerType) -> bool:
        if self == self.__class__.PROTEIN_PROTIEN:
            return p == PolymerType.Protein
        elif self == self.__class__.NUCLEICACID_NUCLEICACID:
            return p == PolymerType.NUCLEICACID_NUCLEICACID
        elif self == self.__class__.NUCLEICACID_PROTEIN:
            return (p == PolymerType.PROTEIN) or (p == PolymerType.NUCLEIC_ACID)
        elif self == self.__class__.LIGAND_PROTEIN:
            return (p == PolymerType.PROTEIN) or (p == PolymerType.LIGAND)
        elif self == self.__class__.LIGAND_NUCLEICACID:
            return (p == PolymerType.NUCLEIC_ACID) or (p == PolymerType.LIGAND)
        elif self == self.__class__.NUCLEICACID_NUCLEICACID:
            return (p == PolymerType.NUCLEICACID_NUCLEICACID)
        else:
            raise NotImplementedError


# Type annotations.
class MolecularInterface(TypedDict):
    asym_id_a: int
    asym_id_b: int
    atom_in_interface: Bool["m"]  # type: ignore


# Supporting utils.
lddt_calc_module = ComputeModelSelectionScore()
rigid_align_calc_module = WeightedRigidAlign()


def create_eligible_asym_id_to_polymer_type_map(
    token_idx_to_asym_ids_map: Int["n"],  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"],  # type: ignore
):
    asym_id_to_polymer_type: Dict[str, PolymerType] = {}
    distinct_asym_ids = set([x.item() for x in token_idx_to_asym_ids_map.unique()])

    for asym_id in distinct_asym_ids:
        is_token_in_asym_id = token_idx_to_asym_ids_map == asym_id
        polymer_types_by_token = is_molecule_types[is_token_in_asym_id].sum(axis=0)
        polymer_type_mol_idx = torch.argmax(polymer_types_by_token.sum(axis=0)).item()
        asym_id_polymer_type = PolymerType.from_molecular_idx(idx=polymer_type_mol_idx)
        asym_id_to_polymer_type[asym_id] = asym_id_polymer_type

    return asym_id_to_polymer_type


def group_identical_entity_asym_ids(
    token_idx_to_asym_ids_map: Int["n"],  # type: ignore
    token_idx_to_entity_ids_map: Int["n"],  # type: ignore
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


def exhaustive_search_for_optimal_asym_id_assignments(
    entity_group_asym_ids: Tuple[int],
    atom_to_asym_id_map: Bool["m"],  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"],  # type: ignore
    predicted_atom_pos: Float["m 3"],  # type: ignore
    true_atom_pos: Float["m 3"],  # type: ignore
):
    """ Exhaustive search of all possible identical asym_id swaps to maximizes lddt. """

    # Identify all permutations of the ordering of the listed asym ids.
    entity_group_asym_id_permutations = itertools.permutations(entity_group_asym_ids)
    max_lddt = float('-inf')
    best_permutation = None
    best_predicted_atom_pos = None

    # Iterate through each permutation of possible asym_id swaps to find the one
    # that maximizes the lddt of the complex.
    for entity_group_asym_id_permutation in entity_group_asym_id_permutations:

        permutations_predicted_atom_pos = torch.clone(predicted_atom_pos)

        # Create a new predicted atom pos tensor by performing the asym id pos swaps.
        for idx, swap_asym_id in enumerate(entity_group_asym_id_permutation):

            # The asym id to be swapped with.
            base_asym_id = entity_group_asym_ids[idx]

            # Skip self-swaps.
            if base_asym_id == swap_asym_id:
                continue

            # Isolate the atomic idxs to swap values for.
            original_atom_position_idxs = atom_to_asym_id_map == base_asym_id
            swap_atom_position_idxs = atom_to_asym_id_map == swap_asym_id

            # Ensure the number of atoms composing the two asym ids are identical. If 
            # they're not, something went wrong.
            if original_atom_position_idxs.sum() != swap_atom_position_idxs.sum():
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
        lddt = lddt_calc_module.compute_lddt(
            true_coords=true_atom_pos.unsqueeze(0),
            pred_coords=permutations_predicted_atom_pos.unsqueeze(0),
            is_dna=is_molecule_types[..., IS_DNA_INDEX].unsqueeze(0),
            is_rna=is_molecule_types[..., IS_RNA_INDEX].unsqueeze(0),
            pairwise_mask=(torch.ones(len(true_atom_pos), len(true_atom_pos)) * True).unsqueeze(0),
        )
        if lddt.item() > max_lddt:
            best_predicted_atom_pos = permutations_predicted_atom_pos
            best_permutation = entity_group_asym_id_permutation
            max_lddt = lddt.item()

    return best_predicted_atom_pos, best_permutation, max_lddt


def sa_search_for_optimal_asym_id_assignments(
    entity_group_asym_ids: Tuple[int],
    atom_to_asym_id_map: Int["m"],  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"],  # type: ignore
    predicted_atom_pos: Float["m 3"],  # type: ignore
    true_atom_pos: Float["m 3"],  # type: ignore
    starting_temp: float = 0.25,
    cooling_rate: float = 0.99,
    num_steps: int = 50,
):
    """ Simulated annealing search for identical asym_id swaps that maximize lddt. """

    temp: float = starting_temp
    iter_num: int = 0


    cstate: Tuple[int] = entity_group_asym_ids
    cstate_atom_pos = torch.clone(predicted_atom_pos)
    
    cstate_lddt = lddt_calc_module.compute_lddt(
        true_coords=true_atom_pos.unsqueeze(0),
        pred_coords=predicted_atom_pos.unsqueeze(0),
        is_dna=is_molecule_types[..., IS_DNA_INDEX].unsqueeze(0),
        is_rna=is_molecule_types[..., IS_RNA_INDEX].unsqueeze(0),
        pairwise_mask=(torch.ones(len(true_atom_pos), len(true_atom_pos)) * True).unsqueeze(0),
    )

    while iter_num < num_steps:
        iter_num += 1

        neighboring_assignment_predicted_atom_pos = torch.clone(predicted_atom_pos)
        nstate = list(cstate)

        # Generate a 'neighboring' state by swapping two asym_ids.
        swap_idxs = random.sample(range(0, len(cstate)), 2)
        nstate[swap_idxs[0]] = cstate[swap_idxs[1]]
        nstate[swap_idxs[1]] = cstate[swap_idxs[0]]


        # Create a new predicted atom pos tensor by performing the asym id pos swaps.
        for idx, swap_asym_id in enumerate(nstate):

            # The asym id to be swapped with.
            base_asym_id = entity_group_asym_ids[idx]

            # Skip self-swaps.
            if base_asym_id == swap_asym_id:
                continue

            # Isolate the atomic idxs to swap values for.
            original_atom_position_idxs = atom_to_asym_id_map == base_asym_id
            swap_atom_position_idxs = atom_to_asym_id_map == swap_asym_id

            # Ensure the number of atoms composing the two asym ids are identical. If 
            # they're not, something went wrong.
            if original_atom_position_idxs.sum() != swap_atom_position_idxs.sum():
                raise ValueError(
                    "Something went wrong, when trying to swap asym_ids to find the "
                    "closest matching assignment to the ground truth, the num atoms "
                    "beloinging to both asym ids do not match. Thus they must not be "
                    f"identical. Found shapes [{original_atom_position_idxs.size(0)}] "
                    f"and [{swap_atom_position_idxs.size(0)}]."
                )

            # Perform the swap on a copy of the predictions.
            neighboring_assignment_predicted_atom_pos[original_atom_position_idxs] = (
                predicted_atom_pos[swap_atom_position_idxs]
            )
            neighboring_assignment_predicted_atom_pos[swap_atom_position_idxs] = (
                predicted_atom_pos[original_atom_position_idxs]
            )

        # Caluclate the lddt between the positions of the actual and the new
        # neighboring swap assignment.
        nstate_lddt = lddt_calc_module.compute_lddt(
            # We don't need different masks for the same asym id.
            true_coords=true_atom_pos.unsqueeze(0),
            pred_coords=neighboring_assignment_predicted_atom_pos.unsqueeze(0),
            is_dna= is_molecule_types[..., IS_DNA_INDEX].unsqueeze(0),
            is_rna=is_molecule_types[..., IS_RNA_INDEX].unsqueeze(0),
            pairwise_mask=(torch.ones(len(true_atom_pos), len(true_atom_pos)) * True).unsqueeze(0),
        )

        # Probabilistically accept a new state.
        acceptance_threshold = (math.e)**((nstate_lddt - cstate_lddt) / temp)
        if random.random() < acceptance_threshold:
            cstate = nstate
            cstate_lddt = nstate_lddt

        # Update the temperature.
        temp = temp * cooling_rate**(iter_num)

    return neighboring_assignment_predicted_atom_pos, cstate, cstate_lddt 


def find_interface_atoms_in_complex(
    atom_pos: Float["m 3"],  # type: ignore
    atom_to_asym_id_map: Int["m"],  # type: ignore
    asym_id_candidate_filter: Optional[Set[int]] = None,
    interface_threshold: float = 30
) -> Dict[str, MolecularInterface]:

    # Validate the atom_pos and atom_to_asym_id_map args have the expected matching dims.
    if atom_pos.size(0) != atom_to_asym_id_map.size(0):
        raise ValueError(
            "Expected the atom_pos and atom_to_asym_id_map args to be tensors of "
            f"identical size, but got [{atom_pos.size(0)}] and [{atom_to_asym_id_map.size(0)}]"
        )

    # Init some data structures for later.
    interfaces = {}
    distinct_asym_ids = set([x.item() for x in atom_to_asym_id_map.unique()])
    eligible_asym_ids = (
        asym_id_candidate_filter 
        if asym_id_candidate_filter is not None else 
        distinct_asym_ids
    )

    # Loop through the (n x m)/2 asym id pairs looking for interface points.
    for asym_id_a in eligible_asym_ids:
        for asym_id_b in eligible_asym_ids:

            # Can't make an interface with yourself -- don't do self comparisons.
            if asym_id_a == asym_id_b:
                continue

            # Create a unique order-agnostic identifier for the interface between
            # two asym ids.
            interface_key = tuple(sorted([asym_id_a, asym_id_b]))

            # Don't want multiple representations of the same interface.
            if interface_key in interfaces:
                continue

            # Get the atom positions of the two asym ids.
            a_pos = atom_pos[atom_to_asym_id_map == asym_id_a]
            b_pos = atom_pos[atom_to_asym_id_map == asym_id_b]

            # Calculate the pairwise distances between atoms in each asym id. This creates a
            # n_atoms_in_asym_a x n_atoms_in_asym_b tensor where the values are the matching
            # atomic distances.
            pairwise_dists = torch.norm(a_pos.unsqueeze(0) - b_pos.unsqueeze(1), dim=2)

            # Bool mask of the atoms that are in interface with eachother.
            is_interface_pair = pairwise_dists < interface_threshold

            # Agg the mask up to determine which atoms are in interface for each asym id.
            asym_a_is_interface_atom_map = torch.any(is_interface_pair, dim=0)
            asym_b_is_interface_atom_map = torch.any(is_interface_pair, dim=1)

            # Create a mapping for each of the input atoms to this fn that indicates whether 
            # the atom is part of an interface.
            is_atom_in_a_interface_stucture = torch.zeros_like(atom_to_asym_id_map, dtype=torch.bool)
            is_atom_in_b_interface_stucture = torch.zeros_like(atom_to_asym_id_map, dtype=torch.bool)

            if len(asym_a_is_interface_atom_map) > 0 and len(asym_b_is_interface_atom_map) > 0:
                is_atom_in_a_interface_stucture[atom_to_asym_id_map == asym_id_a] = asym_a_is_interface_atom_map
                is_atom_in_b_interface_stucture[atom_to_asym_id_map == asym_id_b] = asym_b_is_interface_atom_map
                is_atom_in_interface = is_atom_in_a_interface_stucture | is_atom_in_b_interface_stucture

                interfaces[interface_key] = {
                    "asym_id_a": asym_id_a,
                    "asym_id_b": asym_id_b,
                    "atom_in_interface": is_atom_in_interface,
                }
    
    return interfaces


def calculate_pocket_rmsd(
    predicted_pos: Float["n_interface_atoms 3"], # type: ignore
    true_pos: Float["n_interface_atoms 3"],  # type: ignore
    atom_to_asym_id_map: Int["n_interface_atoms"],  # type: ignore
    center_atom_idxs: Int["n_interface_tokens"] | None,  # type: ignore
):

    # Find pocket atoms.
    pocket_interface = find_interface_atoms_in_complex(
        atom_pos=true_pos,
        atom_to_asym_id_map=atom_to_asym_id_map,
        interface_threshold=10,
    )

    if center_atom_idxs is not None:
        # Build a mask for center atoms.
        non_center_atomic_mask = torch.zeros(predicted_pos.size(0), dtype=torch.Bool)
        non_center_atomic_mask[center_atom_idxs] = True

        # Apply the mask to the atoms in the pocket interface.
        pocket_interface["atom_in_interface"] = (
            pocket_interface["atom_in_interface"] & non_center_atomic_mask
        )

    predicted_pos = predicted_pos[pocket_interface["atom_in_interface"]]

    with torch.no_grad():

        # Perform least squares rigid alignment using the pocket atoms.
        aligned_predicted_pos = rigid_align_calc_module(
            pred_coords=predicted_pos[pocket_interface["atom_in_interface"]].unsqueeze(0),
            true_coords=true_pos[pocket_interface["atom_in_interface"]].unsqueeze(0),
        ).squeeze(0)

        # RMSD calculation.
        sq_diff = torch.square(true_pos.flatten() - aligned_predicted_pos.flatten()).sum()
        msd = torch.mean(sq_diff)
        msd = torch.nan_to_num(msd, nan=0)

    return torch.sqrt(msd).item()



# Analogs of methods from lightining/hydra code so eval can be run outside a lightning context.

def prepare_batch_dict(af3: Alphafold3, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the input batch dictionary for the model.

    :param batch_dict: The input batch dictionary.
    :return: The prepared batch dictionary.
    """
    if not af3.has_molecule_mod_embeds:
        batch_dict["is_molecule_mod"] = None
    return batch_dict


# Core benchmarking script.
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
    parser.add_argument(
        '--complex-type-eval-mode',
         dest="complex_type_eval_mode",
        type=str,
        choices=[e.value for e in ComplexType],
        required=True,
        help='Select an option: ' + ', '.join(e.value for e in ComplexType)
    )


    # Parse the user-provided args.
    args = parser.parse_args()

    # What type of complexs are being evaluated? This governs what interfaces
    # are looked for and how their prediction error is calculated.
    complex_type_eval_mode = ComplexType(args.complex_type_eval_mode)

    # The radius to use when calculating interface prediction error metrics.
    inclusion_raidus = (
        30 
        if complex_type_eval_mode == ComplexType.NUCLEICACID_PROTEIN
        else 15
    )

    # Load the model-weights into an Alphafold3 obj.
    alphafold3 = Alphafold3.init_and_load(path=args.inference_model_weights)
    alphafold3.to(DEVICE)
    alphafold3.eval()

    # Validate the alphafold3 dimensions meet the script's expectations.
    if alphafold3.dim_atom_inputs != EXPECTED_DIM_ATOM_INPUT:
        raise ValueError(
            f"The alphafold3 model loaded from [{args.inference_model_weights}] uses a "
            f"atom input shape 'dim_atom_inputs' [{alphafold3.dim_atom_inputs}] that "
            f"differs from what this script expects [{EXPECTED_DIM_ATOM_INPUT}]"
        )

    # Load the pdb file dataset into a reference obj.
    pdb_dataset = PDBDataset(folder=args.pdb_dataset)
    atom_dataset = pdb_dataset_to_atom_inputs(
        pdb_dataset=pdb_dataset,
        return_atom_dataset=True,
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
            #    **batch_dict,
            #    return_loss=False,
            #    return_confidence_head_logits=True,
            #    return_distogram_head_logits=True,
            #)
            batch_sampled_atom_pos = torch.randn(2, 3009, 3)

        for batch_idx in range(batched_atom_input.atom_inputs.size(0)):

            # Pull out tensorized mappings between tensor indices and parent entities
            # for a single complex.
            padded_atom_lens_per_token_idx = batched_atom_input.molecule_atom_lens[batch_idx]
            padded_token_idx_to_center_atom_idx = batched_atom_input.molecule_atom_indices[batch_idx]
            padded_token_idx_to_is_molecule_types = batched_atom_input.is_molecule_types[batch_idx]
            _, _, padded_token_idx_to_asym_ids, padded_token_idx_to_entity_id, _= (
                batched_atom_input.additional_molecule_feats[batch_idx].unbind(dim=-1)
            )

            # Find the unpadded token indices of this complex.
            unpadded_token_idxs = padded_token_idx_to_asym_ids != INT_PAD_VALUE

            # Remove the padding from all the descriptive batched tensors.
            token_idx_to_asym_ids = padded_token_idx_to_asym_ids[unpadded_token_idxs]
            token_idx_to_entity_ids = padded_token_idx_to_entity_id[unpadded_token_idxs]
            token_idx_to_center_atom_idx = padded_token_idx_to_center_atom_idx[unpadded_token_idxs]
            atom_lens_per_token_idx = padded_atom_lens_per_token_idx[unpadded_token_idxs]
            token_idx_to_is_molecule_types = padded_token_idx_to_is_molecule_types[
                unpadded_token_idxs
            ]

            # Find the unpadded atomic indices of this complex.
            unpadded_atom_idxs = torch.arange(0, atom_lens_per_token_idx.sum()).long()

            # Construct unpadded atom level mappings to various atomic attributes.
            atom_to_asym_id_map = token_idx_to_asym_ids.repeat_interleave(
                atom_lens_per_token_idx,
            )
            atom_to_is_molecule_types = token_idx_to_is_molecule_types.repeat_interleave(
                atom_lens_per_token_idx,
                dim=0,
            )

            true_atom_pos = batched_atom_input.atom_pos[batch_idx][unpadded_atom_idxs, ...]
            predicted_atom_pos = batch_sampled_atom_pos[batch_idx][unpadded_atom_idxs, ...]

            # Group all the asym_ids that are "identical" together at the token level.
            identical_entity_asym_id_groups = group_identical_entity_asym_ids(
                token_idx_to_asym_ids_map=token_idx_to_asym_ids,
                token_idx_to_entity_ids_map=token_idx_to_entity_ids
            )

            # For each group of matching asym_ids, swap their positions to find the
            # arrangement that most closely matches the obersrved arrangement.
            for entity_group_asym_ids in identical_entity_asym_id_groups:
                
                if len(entity_group_asym_ids) <= 1:
                    best_predicted_atom_pos = predicted_atom_pos
                    # No swapping arrangements to search through.

                elif len(entity_group_asym_ids) <= MAX_NUM_IDENTICAL_ENTITIES_FOR_EXHAUSTIVE_SEARCH:
                    # Few enough matching entities to do an exhaustive search.
                    best_predicted_atom_pos, best_permutation, max_lddt = (
                        exhaustive_search_for_optimal_asym_id_assignments(
                            entity_group_asym_ids=entity_group_asym_ids,
                            atom_to_asym_id_map=atom_to_asym_id_map,
                            is_molecule_types=atom_to_is_molecule_types,
                            predicted_atom_pos=predicted_atom_pos,
                            true_atom_pos=true_atom_pos,
                        )
                    )

                else:
                    best_predicted_atom_pos, best_permutation, max_lddt = (
                        sa_search_for_optimal_asym_id_assignments(
                            entity_group_asym_ids=entity_group_asym_ids,
                            atom_to_asym_id_map=atom_to_asym_id_map,
                            is_molecule_types=atom_to_is_molecule_types,
                            predicted_atom_pos=predicted_atom_pos,
                            true_atom_pos=true_atom_pos,
                        )
                    )

            #TODO: Ligand symmetry resolution logic?

            # Create a map of each asym id to its polymer type.
            asym_id_to_polymer_type = create_eligible_asym_id_to_polymer_type_map(
                token_idx_to_asym_ids_map=token_idx_to_asym_ids,
                is_molecule_types=token_idx_to_is_molecule_types,
            )

            # Identify the asym_ids that should be evaluated given the complex type 
            # specified in the script args.
            eligible_asym_ids = set([
                k for k, v in asym_id_to_polymer_type.items() 
                if complex_type_eval_mode.is_eligible_polymer(p=v)
            ])

            # Identify all the interfaces in the complex eligible for evaluation.
            interfaces = find_interface_atoms_in_complex(
                atom_pos=batched_atom_input.atom_pos[batch_idx][unpadded_atom_idxs, ...],
                atom_to_asym_id_map=atom_to_asym_id_map,
                asym_id_candidate_filter=eligible_asym_ids,
                interface_threshold=inclusion_raidus,
            )

            # TODO: Remove this? Turn into warning?
            if len(interfaces) == 0:
                print("sus...")

            # Compute interface level metrics for each interface.
            for interface_key in interfaces:
                interface = interfaces[interface_key]
                center_atom_idxs = None

                if complex_type_eval_mode == ComplexType.LIGAND_PROTEIN:
                    center_atom_idxs = token_idx_to_center_atom_idx

                pocket_rmsd = calculate_pocket_rmsd(
                    predicted_pos=best_predicted_atom_pos[interface["atom_in_interface"]],
                    true_pos=true_atom_pos[interface["atom_in_interface"]],
                    atom_to_asym_id_map=atom_to_asym_id_map[interface["atom_in_interface"]],
                    center_atom_idxs=center_atom_idxs,
                )

                ilddt = lddt_calc_module.compute_lddt(
                    true_coords=true_atom_pos[interface["atom_in_interface"]].unsqueeze(0),
                    pred_coords=best_predicted_atom_pos[interface["atom_in_interface"]].unsqueeze(0),
                    is_dna=is_molecule_types[interface["atom_in_interface"]][..., IS_DNA_INDEX].unsqueeze(0),
                    is_rna=is_molecule_types[interface["atom_in_interface"]][..., IS_RNA_INDEX].unsqueeze(0),
                    pairwise_mask=(torch.ones(len(interface["atom_in_interface"]), len(interface["atom_in_interface"])))
                )

                # TODO: DockQ implementation?

            for asym_id in eligible_asym_ids:
                # TODO: What does the paper mean by entity lddt radius of 30Å for nucleic
                # acid entities?
                lddt = lddt_calc_module.compute_lddt(
                    true_coords=true_atom_pos[interface["atom_in_interface"]].unsqueeze(0),
                    pred_coords=best_predicted_atom_pos[interface["atom_in_interface"]].unsqueeze(0),
                    is_dna=is_molecule_types[interface["atom_in_interface"]][..., IS_DNA_INDEX].unsqueeze(0),
                    is_rna=is_molecule_types[interface["atom_in_interface"]][..., IS_RNA_INDEX].unsqueeze(0),
                    pairwise_mask=(torch.ones(len(interface["atom_in_interface"]), len(interface["atom_in_interface"])))
                )









        
