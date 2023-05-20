from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class AminoAcidTokenizer(PreTrainedTokenizer):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        characters: Sequence[str] = "ACDEFGHIKLMNPQRSTVWY",
        extra_id_prefix: str = "extra_id_",
        extra_ids: int = 10,
        additional_special_tokens: list[str] = [],
        **kwargs,
    ) -> None:
        # Add extra_ids to the special token list
        if extra_ids > 0 and not additional_special_tokens:
            additional_special_tokens = [f"<{extra_id_prefix}{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(
                set(
                    filter(
                        lambda x: bool(extra_id_prefix in str(x)),
                        additional_special_tokens,
                    )
                )
            )
            if extra_tokens != extra_ids:
                raise ValueError(  # noqa: E501
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens "
                    f"({additional_special_tokens}) are provided to T5Tokenizer."
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        self.characters = characters
        self.extra_id_prefix = extra_id_prefix
        eos_token = AddedToken("\n", lstrip=False, rstrip=False)
        sep_token = AddedToken("\n", lstrip=False, rstrip=False)
        pad_token = AddedToken(" ", lstrip=False, rstrip=False)
        unk_token = AddedToken("?", lstrip=False, rstrip=False)
        mask_token = AddedToken("*", lstrip=True, rstrip=False)

        super().__init__(
            eos_token=eos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self._vocab_str_to_int = {
            " ": 0,
            "\n": 1,
            "*": 2,
            "?": 3,
            "+": 4,
            "=": 5,
            "-": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        self.unique_no_split_tokens = (
            list(self._vocab_str_to_int.keys()) + additional_special_tokens
        )
        self._create_trie(self.unique_no_split_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int) + len(self.additional_special_tokens)

    def get_sentinel_tokens(self) -> list[str]:
        return sorted(
            set(
                filter(
                    lambda x: bool(re.search(rf"<{self.extra_id_prefix}\d+>", x)) is not None,
                    self.additional_special_tokens,
                )
            )
        )

    def get_sentinel_token_ids(self) -> list[int]:
        return [self._convert_token_to_id(token) for token in self.get_sentinel_tokens()]

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        if token.startswith(f"<{self.extra_id_prefix}"):
            match = re.match(rf"<{self.extra_id_prefix}(\d+)>", token)
            assert match is not None
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["?"])

    def _convert_id_to_token(self, index: int) -> str:
        if index < len(self._vocab_str_to_int):
            token = self._vocab_int_to_str[index]
        else:
            token = f"<{self.extra_id_prefix}{self.vocab_size - 1 - index}>"
        return token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]

        result = len(token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    # @classmethod
    # def from_config(cls, config: Dict) -> AminoAcidTokenizer:
    #     cfg = {}
    #     cfg["characters"] = [chr(i) for i in config["char_ords"]]
    #     cfg["model_max_length"] = config["model_max_length"]
    #     return cls(**cfg)

    # def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
    #     cfg_file = Path(save_directory) / "tokenizer_config.json"
    #     cfg = self.get_config()
    #     with open(cfg_file, "w") as f:
    #         json.dump(cfg, f, indent=4)

    # @classmethod
    # def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
    #     cfg_file = Path(save_directory) / "tokenizer_config.json"
    #     with open(cfg_file) as f:
    #         cfg = json.load(f)
    #     return cls.from_config(cfg)
