from dataclasses import dataclass
from typing import Any, Callable, List, Type

from rdkit.Chem.rdchem import Mol

from alphafold3_pytorch.utils.tensor_typing import Bool, Float, Int, typecheck
from alphafold3_pytorch.utils.utils import exists, identity

# constants

IS_MOLECULE_TYPES = 4
ADDITIONAL_MOLECULE_FEATS = 5

# functions


def compose(*fns: Callable):
    """A function for chaining from Alphafold3Input -> MoleculeInput -> AtomInput."""

    def inner(x, *args, **kwargs):
        """Inner function for chaining the functions together."""
        for fn in fns:
            x = fn(x, *args, **kwargs)
        return x

    return inner


# atom level, what Alphafold3 accepts


@typecheck
@dataclass
class AtomInput:
    atom_inputs: Float["m dai"]  # type: ignore
    molecule_ids: Int[" n"]  # type: ignore
    molecule_atom_lens: Int[" n"]  # type: ignore
    atompair_inputs: Float["m m dapi"] | Float["nw w (w*2) dapi"]  # type: ignore
    additional_molecule_feats: Float[f"n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"]  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dm"] | None = None  # type: ignore
    token_bonds: Bool["n n"] | None = None  # type: ignore
    atom_ids: Int[" m"] | None = None  # type: ignore
    atom_parent_ids: Int[" m"] | None = None  # type: ignore
    atompair_ids: Int["m m"] | Int["nw w (w*2)"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    atom_pos: Float["m 3"] | None = None  # type: ignore
    molecule_atom_indices: Int[" n"] | None = None  # type: ignore
    distance_labels: Int["n n"] | None = None  # type: ignore
    pae_labels: Int["n n"] | None = None  # type: ignore
    pde_labels: Int["n n"] | None = None  # type: ignore
    plddt_labels: Int[" n"] | None = None  # type: ignore
    resolved_labels: Int[" n"] | None = None  # type: ignore


@typecheck
@dataclass
class BatchedAtomInput:
    atom_inputs: Float["b m dai"]  # type: ignore
    molecule_ids: Int["b n"]  # type: ignore
    molecule_atom_lens: Int["b n"]  # type: ignore
    atompair_inputs: Float["b m m dapi"] | Float["b nw w (w*2) dapi"]  # type: ignore
    additional_molecule_feats: Float[f"b n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"]  # type: ignore
    templates: Float["b t n n dt"] | None = None  # type: ignore
    msa: Float["b s n dm"] | None = None  # type: ignore
    token_bonds: Bool["b n n"] | None = None  # type: ignore
    atom_ids: Int["b m"] | None = None  # type: ignore
    atom_parent_ids: Int["b m"] | None = None  # type: ignore
    atompair_ids: Int["b m m"] | Int["b nw w (w*2)"] | None = None  # type: ignore
    template_mask: Bool["b t"] | None = None  # type: ignore
    msa_mask: Bool["b s"] | None = None  # type: ignore
    atom_pos: Float["b m 3"] | None = None  # type: ignore
    molecule_atom_indices: Int["b n"] | None = None  # type: ignore
    distance_labels: Int["b n n"] | None = None  # type: ignore
    pae_labels: Int["b n n"] | None = None  # type: ignore
    pde_labels: Int["b n n"] | None = None  # type: ignore
    plddt_labels: Int["b n"] | None = None  # type: ignore
    resolved_labels: Int["b n"] | None = None  # type: ignore


# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens


@typecheck
@dataclass
class MoleculeInput:
    molecules: List[Mol]
    molecule_token_pool_lens: List[List[int]]
    molecule_atom_indices: List[List[int] | None]
    molecule_ids: Int[" n"]  # type: ignore
    additional_molecule_feats: Float[f"n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"]  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dm"] | None = None  # type: ignore
    atom_pos: List[Float["_ 3"]] | Float["m 3"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    distance_labels: Int["n n"] | None = None  # type: ignore
    pae_labels: Int["n n"] | None = None  # type: ignore
    pde_labels: Int[" n"] | None = None  # type: ignore
    resolved_labels: Int[" n"] | None = None  # type: ignore


@typecheck
def molecule_to_atom_input(molecule_input: MoleculeInput) -> AtomInput:
    """Converts a MoleculeInput to an AtomInput."""
    raise NotImplementedError


# alphafold3 input - support polypeptides, nucleic acids, metal ions + any number of ligands + misc biomolecules


@typecheck
@dataclass
class Alphafold3Input:
    proteins: List[Int[" _"] | str]  # type: ignore
    ss_dna: List[Int[" _"] | str]  # type: ignore
    ss_rna: List[Int[" _"] | str]  # type: ignore
    metal_ions: Int[" _"] | List[str]  # type: ignore
    misc_molecule_ids: Int[" _"] | List[str]  # type: ignore
    ligands: List[Mol | str]  # can be given as smiles
    ds_dna: List[Int[" _"] | str]  # type: ignore
    ds_rna: List[Int[" _"] | str]  # type: ignore
    templates: Float["t n n dt"] | None = None  # type: ignore
    msa: Float["s n dm"] | None = None  # type: ignore
    atom_pos: List[Float["_ 3"]] | Float["m 3"] | None = None  # type: ignore
    template_mask: Bool[" t"] | None = None  # type: ignore
    msa_mask: Bool[" s"] | None = None  # type: ignore
    distance_labels: Int["n n"] | None = None  # type: ignore
    pae_labels: Int["n n"] | None = None  # type: ignore
    pde_labels: Int[" n"] | None = None  # type: ignore
    resolved_labels: Int[" n"] | None = None  # type: ignore


@typecheck
def alphafold3_input_to_molecule_input(alphafold3_input: Alphafold3Input) -> MoleculeInput:
    """Converts an Alphafold3Input to a MoleculeInput."""
    raise NotImplementedError


# pdb input


@typecheck
@dataclass
class PDBInput:
    filepath: str


@typecheck
def pdb_input_to_alphafold3_input(pdb_input: PDBInput) -> Alphafold3Input:
    """Converts a PDBInput to an Alphafold3Input."""
    raise NotImplementedError


# the config used for keeping track of all the disparate inputs and their transforms down to AtomInput
# this can be preprocessed or will be taken care of automatically within the Trainer during data collation

INPUT_TO_ATOM_TRANSFORM = {
    AtomInput: identity,
    MoleculeInput: molecule_to_atom_input,
    Alphafold3Input: compose(alphafold3_input_to_molecule_input, molecule_to_atom_input),
    PDBInput: compose(
        pdb_input_to_alphafold3_input, alphafold3_input_to_molecule_input, molecule_to_atom_input
    ),
}

# function for extending the config


@typecheck
def register_input_transform(input_type: Type, fn: Callable[[Any], AtomInput]):
    """Registers a new input transform function."""
    assert input_type not in INPUT_TO_ATOM_TRANSFORM, f"{input_type} is already registered"
    INPUT_TO_ATOM_TRANSFORM[input_type] = fn


# functions for transforming to atom inputs


@typecheck
def maybe_transform_to_atom_input(i: Any) -> AtomInput:
    """Transforms a list of inputs to AtomInputs."""
    maybe_to_atom_fn = INPUT_TO_ATOM_TRANSFORM.get(type(i), None)

    if not exists(maybe_to_atom_fn):
        raise TypeError(
            f"Invalid input type {type(i)} being passed into Trainer that is not converted to AtomInput correctly"
        )

    return maybe_to_atom_fn(i)


@typecheck
def maybe_transform_to_atom_inputs(inputs: List[Any]) -> List[AtomInput]:
    """Transforms a list of inputs to AtomInputs."""
    return [maybe_transform_to_atom_input(i) for i in inputs]
