"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN


class Vocabulary:
    _shared_vocab: Dict[str, int] = {}
    _shared_id_to_token: Dict[int, str] = {}

    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        self.pad_token_id: int = 0
        self.unk_token_id: int = 1
        self.bos_token_id: int = 50256
        self.eos_token_id: int = 50256

        self.add_token(PAD_TOKEN, 0)
        self.add_token(UNK_TOKEN, 1)
        self.add_token(BOS_TOKEN, self.bos_token_id)

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)
        else:
            for token, idx in Vocabulary._shared_vocab.items():
                self.add_token(token, idx)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = len(self._token_to_id)
            while idx in self._id_to_token:
                idx += 1
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        Vocabulary._shared_vocab[token] = idx
        Vocabulary._shared_id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if token in Vocabulary._shared_vocab:
            idx = Vocabulary._shared_vocab[token]
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
            return idx
        return self.add_token(token)

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        if idx in Vocabulary._shared_id_to_token:
            return Vocabulary._shared_id_to_token[idx]
        return f"<token_{idx}>"

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id or token in Vocabulary._shared_vocab
