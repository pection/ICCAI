from typing import Dict, Union


class ClosureVocab:
    """
    Simple class to wrap CLOSURE's ugly vocab dictionary
    """

    def __init__(self, vocab: Dict[str, Union[Dict[str, int], Dict[int, str]]]):
        super().__init__()
        self.vocab = vocab

    # backwards compatibility with "just a dict"
    def __getitem__(self, item) -> Union[Dict[str, int], Dict[int, str]]:
        return self.vocab[item]

    def question_token_to_idx(self, token: str) -> int:
        return self.vocab["question_token_to_idx"][token]

    def program_token_to_idx(self, token: str) -> int:
        return self.vocab["program_token_to_idx"][token]

    def answer_token_to_idx(self, token: str) -> int:
        return self.vocab["answer_token_to_idx"][token]

    def question_idx_to_token(self, idx: int) -> str:
        return self.vocab["question_idx_to_token"][idx]

    def program_idx_to_token(self, idx: int) -> str:
        return self.vocab["program_idx_to_token"][idx]

    def answer_idx_to_token(self, idx: int) -> str:
        return self.vocab["answer_idx_to_token"][idx]
