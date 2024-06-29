from typing import Callable, List, Literal, Type, TypedDict

from rdkit.Chem.rdchem import Mol

from alphafold3_pytorch.utils.tensor_typing import Bool, Float, Int, typecheck

# constants

IS_MOLECULE_TYPES = 4
ADDITIONAL_MOLECULE_FEATS = 5

# atom level, what AlphaFold3 accepts


@typecheck
class AtomInput(TypedDict):
    atom_inputs: Float["m dai"]  # type: ignore
    molecule_ids: Int["n"]  # type: ignore
    molecule_atom_lens: Int["n"]  # type: ignore
    atompair_inputs: Float["m m dapi"] | Float["nw w (w*2) dapi"]  # type: ignore
    additional_molecule_feats: Float[f"n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"n {IS_MOLECULE_TYPES}"]  # type: ignore
    templates: Float["t n n dt"]  # type: ignore
    msa: Float["s n dm"]  # type: ignore
    token_bonds: Bool["n n"] | None  # type: ignore
    atom_ids: Int["m"] | None  # type: ignore
    atom_parent_ids: Int["m"] | None  # type: ignore
    atompair_ids: Int["m m"] | Int["nw w (w*2)"] | None  # type: ignore
    template_mask: Bool["t"] | None  # type: ignore
    msa_mask: Bool["s"] | None  # type: ignore
    atom_pos: Float["m 3"] | None  # type: ignore
    molecule_atom_indices: Int["n"] | None  # type: ignore
    distance_labels: Int["n n"] | None  # type: ignore
    pae_labels: Int["n n"] | None  # type: ignore
    pde_labels: Int["n"] | None  # type: ignore
    resolved_labels: Int["n"] | None  # type: ignore


@typecheck
class BatchedAtomInput(TypedDict):
    atom_inputs: Float["b m dai"]  # type: ignore
    molecule_ids: Int["b n"]  # type: ignore
    molecule_atom_lens: Int["b n"]  # type: ignore
    atompair_inputs: Float["b m m dapi"] | Float["b nw w (w*2) dapi"]  # type: ignore
    additional_molecule_feats: Float[f"b n {ADDITIONAL_MOLECULE_FEATS}"]  # type: ignore
    is_molecule_types: Bool[f"b n {IS_MOLECULE_TYPES}"]  # type: ignore
    templates: Float["b t n n dt"]  # type: ignore
    msa: Float["b s n dm"]  # type: ignore
    token_bonds: Bool["b n n"] | None  # type: ignore
    atom_ids: Int["b m"] | None  # type: ignore
    atom_parent_ids: Int["b m"] | None  # type: ignore
    atompair_ids: Int["b m m"] | Int["b nw w (w*2)"] | None  # type: ignore
    template_mask: Bool["b t"] | None  # type: ignore
    msa_mask: Bool["b s"] | None  # type: ignore
    atom_pos: Float["b m 3"] | None  # type: ignore
    molecule_atom_indices: Int["b n"] | None  # type: ignore
    distance_labels: Int["b n n"] | None  # type: ignore
    pae_labels: Int["b n n"] | None  # type: ignore
    pde_labels: Int["b n"] | None  # type: ignore
    resolved_labels: Int["b n"] | None  # type: ignore


# molecule input - accepting list of molecules as rdchem.Mol + the atomic lengths for how to pool into tokens


@typecheck
class MoleculeInput(TypedDict):
    molecules: List[Mol]
    molecule_token_pool_lens: List[List[int]]
    molecule_atom_indices: List[List[int] | None]
    molecule_ids: Int["n"]  # type: ignore
    additional_molecule_feats: Float["n 9"]  # type: ignore
    atom_pos: Float["m 3"] | None  # type: ignore
    templates: Float["t n n dt"]  # type: ignore
    template_mask: Bool["t"] | None  # type: ignore
    msa: Float["s n dm"]  # type: ignore
    msa_mask: Bool["s"] | None  # type: ignore
    distance_labels: Int["n n"] | None  # type: ignore
    pae_labels: Int["n n"] | None  # type: ignore
    pde_labels: Int["n"] | None  # type: ignore
    resolved_labels: Int["n"] | None  # type: ignore


@typecheck
def molecule_to_atom_input(molecule_input: MoleculeInput) -> AtomInput:
    """Converts a MoleculeInput to an AtomInput."""
    raise NotImplementedError


def validate_molecule_input(molecule_input: MoleculeInput):
    """Validates a MoleculeInput."""
    assert True


# residue level - single chain proteins for starters


@typecheck
class SingleProteinInput(TypedDict):
    residue_ids: Int["n"]  # type: ignore
    residue_atom_lens: Int["n"]  # type: ignore
    templates: Float["t n n dt"]  # type: ignore
    msa: Float["s n dm"]  # type: ignore
    template_mask: Bool["t"] | None  # type: ignore
    msa_mask: Bool["s"] | None  # type: ignore
    atom_pos: Float["m 3"] | None  # type: ignore
    distance_labels: Int["n n"] | None  # type: ignore
    pae_labels: Int["n n"] | None  # type: ignore
    pde_labels: Int["n"] | None  # type: ignore
    resolved_labels: Int["n"] | None  # type: ignore


@typecheck
def single_protein_input_to_atom_input(input: SingleProteinInput) -> AtomInput:
    """Converts a SingleProteinInput to an AtomInput."""
    raise NotImplementedError


# single chain protein with single ds nucleic acid

# o - for nucleOtide seq


@typecheck
class SingleProteinSingleNucleicAcidInput(TypedDict):
    residue_ids: Int["n"]  # type: ignore
    residue_atom_lens: Int["n"]  # type: ignore
    nucleotide_ids: Int["o"]  # type: ignore
    nucleic_acid_type: Literal["dna", "rna"]  # type: ignore
    templates: Float["t n n dt"]  # type: ignore
    msa: Float["s n dm"]  # type: ignore
    template_mask: Bool["t"] | None  # type: ignore
    msa_mask: Bool["s"] | None  # type: ignore
    atom_pos: Float["m 3"] | None  # type: ignore
    distance_labels: Int["n n"] | None  # type: ignore
    pae_labels: Int["n n"] | None  # type: ignore
    pde_labels: Int["n"] | None  # type: ignore
    resolved_labels: Int["n"] | None  # type: ignore


@typecheck
def single_protein_input_and_single_nucleic_acid_to_atom_input(
    input: SingleProteinSingleNucleicAcidInput,
) -> AtomInput:
    """Converts a SingleProteinSingleNucleicAcidInput to an AtomInput."""
    raise NotImplementedError


# the config used for keeping track of all the disparate inputs and their transforms down to AtomInput
# this can be preprocessed or will be taken care of automatically within the Trainer during data collation

INPUT_TO_ATOM_TRANSFORM = {
    MoleculeInput: molecule_to_atom_input,
    SingleProteinInput: single_protein_input_to_atom_input,
    SingleProteinSingleNucleicAcidInput: single_protein_input_and_single_nucleic_acid_to_atom_input,
}

# function for extending the config


@typecheck
def register_input_transform(input_type: Type, fn: Callable[[TypedDict], AtomInput]):  # type: ignore
    """Registers a new input transform."""
    INPUT_TO_ATOM_TRANSFORM[input_type] = fn
