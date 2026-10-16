from pathlib import Path
from typing import Any, Union
from .vocab_parser import VocabParser
from .models import FunctionDefinition

class ConstrainedDecoder:
    """Enforces JSON schema constraints on LLM decoding using logit masking."""
    def __init__(self, model: Any, vocab_file_path: Path, function_definitions: list[FunctionDefinition],) -> None:
        """Initialize ConstrainedDecoder with model, vocabulary, and function schemas."""
        self.model = model
        self.vocab_parser = VocabParser(vocab_file_path)
        self.function_definitions = function_definitions
        self.func_map = {fn.name: fn for fn in function_definitions}


    def _apply_mask(self, logits: list[float], allowed_token_ids: set[int]) -> list[float]:
        """Apply logit masking by setting non-allowed token scores to -inf.
        Args:
            logits: Raw unconstrained logit scores from the LLM.
            allowed_token_ids: Set of token IDs permitted by current JSON state.
        Returns: The masked logits list where disallowed tokens are -inf. """
        if not allowed_token_ids:
            raise ValueError("No allowed token IDs provided for logit masking.")
        for i in range(len(logits)):
            if i not in allowed_token_ids:
                logits[i] = float('-inf')
        return logits
    
    def _get_allowed_token_ids():
        ...
