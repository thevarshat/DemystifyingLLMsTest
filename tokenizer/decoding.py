"""Token decoding."""
from typing import List, Optional
from .vocab import Vocabulary
from .special_tokens import SPECIAL_TOKENS


def decode(ids: List[int], vocab: Optional[Vocabulary] = None, skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs back to a string.
    """
    if vocab is None:
        vocab = Vocabulary()

    tokens = [vocab.id_to_token(i) for i in ids]

    if skip_special_tokens:
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]

    return " ".join(tokens)
