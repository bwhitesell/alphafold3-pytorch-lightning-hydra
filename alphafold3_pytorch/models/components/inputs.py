from typing import TypedDict

from alphafold3_pytorch.utils.typing import Bool, Float, Int, typecheck

# constants


@typecheck
class AtomInput(TypedDict):
    """A collection of inputs to AlphaFold 3."""

    atom_inputs: Float["m dai"]  # type: ignore
    molecule_atom_lens: Int[" n"]  # type: ignore
    atompair_inputs: Float["m m dapi"] | Float["nw w (w*2) dapi"]  # type: ignore
    additional_molecule_feats: Float["n 10"]  # type: ignore
    templates: Float["t n n dt"]  # type: ignore
    msa: Float["s n dm"]  # type: ignore
    template_mask: Bool[" t"] | None  # type: ignore
    msa_mask: Bool[" s"] | None  # type: ignore
    atom_pos: Float["m 3"] | None  # type: ignore
    molecule_atom_indices: Int[" n"] | None  # type: ignore
    distance_labels: Int["n n"] | None  # type: ignore
    pae_labels: Int["n n"] | None  # type: ignore
    pde_labels: Int[" n"] | None  # type: ignore
    resolved_labels: Int[" n"] | None  # type: ignore
