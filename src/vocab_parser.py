import json
from pathlib import Path
from typing import Dict, Set, Union

class VocabParser:
    """Parses model vocabulary and maps token IDs to string representations.
    Handles token-to-string decoding, vocabulary loading, and prefix matching
    for constrained decoding state transitions.
    """

    def __init__(self, vocab_file_path: Union[str, Path]) -> None:
        """Initialize VocabParser by loading the vocabulary JSON file.
        Args:
            vocab_file_path: Path to the vocabulary file from Small_LLM_Model.
        """
        self.vocab_file_path = Path(vocab_file_path)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.cleaned_id_to_token: Dict[int, str] = {}
        self._load_vocab()

    def _load_vocab(self) -> None:
        """Load vocabulary file and build raw and cleaned mapping dictionaries."""
        try:
            with self.vocab_file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found at: {self.vocab_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in vocab file {self.vocab_file_path}: {e}")

        if isinstance(data, dict):
            for token, idx in data.items():
                if isinstance(idx, int):
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token
                    self.cleaned_id_to_token[idx] = self._clean_token(token)
        else:
            raise ValueError("Vocab file must be a JSON object (str -> int).")

    @staticmethod
    def _clean_token(token: str) -> str:
        """Clean specific subword whitespace symbol 'Ġ' or '▁'."""
        return token.replace("Ġ", " ").replace("▁")

    def get_token_str(self, token_id: int) -> str:
        """Get the string representation for a given token ID."""
        return self.id_to_token.get(token_id, "")

    def get_matching_token_ids(self, allowed_tokens: Set[str], prefix: str = "") -> Set[int]:
        """Find token IDs matching allowed continuations or prefixes.
        Args:
            allowed_tokens: The set of exact target strings allowed.
            prefix: Partial string already emitted in the current state.
        Returns:
            Set of valid token IDs that can be legally sampled next.
        """
        valid_ids: Set[int] = set()
        for token_id, token_str in self.cleaned_id_to_token.items():
            if not token_str:
                continue
            for target in allowed_tokens:
                remainder = target[len(prefix):] if prefix else target
                if remainder.startswith(token_str) or token_str.startswith(remainder):
                    valid_ids.add(token_id)
                    break
        return valid_ids

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return len(self.id_to_token)
