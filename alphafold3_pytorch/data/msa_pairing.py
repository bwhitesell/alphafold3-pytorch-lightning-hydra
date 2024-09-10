import numpy as np
from beartype.typing import Dict, List
from torch import Tensor

from alphafold3_pytorch.utils.tensor_typing import typecheck


@typecheck
def create_paired_features(
    chains: List[Dict[str, Tensor | np.ndarray]]
) -> List[Dict[str, Tensor | np.ndarray]]:
    """
    Pair MSA chain features.
    From:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/msa_pairing.py#L56

    :param chains: The MSA chain feature dictionaries.
    :return: The MSA chain feature dictionaries with paired features.
    """
    chain_keys = chains[0].keys()
    return chains


@typecheck
def deduplicate_unpaired_sequences(
    chains: List[Dict[str, Tensor | np.ndarray]]
) -> List[Dict[str, Tensor | np.ndarray]]:
    """
    Deduplicate unpaired chain sequences.
    From:
    https://github.com/aqlaboratory/openfold/blob/6f63267114435f94ac0604b6d89e82ef45d94484/openfold/data/msa_pairing.py#L462

    :param chains: The MSA chain feature dictionaries.
    :return: The MSA chain feature dictionaries with deduplicated unpaired sequences.
    """
    return chains
