from __future__ import annotations

from functools import partial

import numpy as np
import rootutils
import torch
from beartype import beartype
from beartype.door import is_bearable
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import DisorderedResidue, Residue
from environs import Env
from jaxtyping import Bool, Float, Int, Shaped, jaxtyped
from torch import Tensor

from alphafold3_pytorch.utils.model_utils import package_available
from alphafold3_pytorch.utils.utils import always, identity

# environment

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

env = Env()
env.read_env()


# NOTE: `jaxtyping` is a misnomer, works for PyTorch as well


class TorchTyping:
    """Torch typing."""

    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """Get item."""
        return self.abstract_dtype[Tensor, shapes]


Shaped = TorchTyping(Shaped)
Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)

# helper type aliases

IntType = int | np.int32 | np.int64
AtomType = Atom | DisorderedAtom
ResidueType = Residue | DisorderedResidue
ChainType = Chain
TokenType = AtomType | ResidueType

# some more colocated environmental stuff

# maybe deespeed checkpoint, and always use non reentrant checkpointing

DEEPSPEED_CHECKPOINTING = env.bool("DEEPSPEED_CHECKPOINTING", False)

if DEEPSPEED_CHECKPOINTING:
    assert package_available("deepspeed"), "DeepSpeed must be installed for checkpointing."

    import deepspeed

    checkpoint = deepspeed.checkpointing.checkpoint
else:
    checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

# NOTE: use env variable `TYPECHECK` (which is set by `rootutils` above using `.env`) to control whether to use `beartype` + `jaxtyping`
# NOTE: use env variable `DEBUG` to control whether to print debugging information

should_typecheck = env.bool("TYPECHECK", False)
IS_DEBUGGING = env.bool("DEBUG", False)

typecheck = jaxtyped(typechecker=beartype) if should_typecheck else identity

beartype_isinstance = is_bearable if should_typecheck else always(True)

if should_typecheck:
    print("Type checking is enabled.")
else:
    print("Type checking is disabled.")

if IS_DEBUGGING:
    print("Debugging is enabled.")
else:
    print("Debugging is disabled.")

__all__ = [
    beartype_isinstance,
    Bool,
    Float,
    Int,
    Shaped,
    checkpoint,
    should_typecheck,
    typecheck,
    IS_DEBUGGING,
]
