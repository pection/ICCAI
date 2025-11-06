from vqa_framework.models.symbolic_exec import SymbolicShapesExecutor
import pytest
import copy
import torch
from typing import List
from vqa_framework.utils.vocab import ClosureVocab

EXAMPLE_VOCAB = {
    "question_token_to_idx": {
        "<NULL>": 0,
        "<START>": 1,
        "<END>": 2,
        "<UNK>": 3,
        "a": 4,
        "above": 5,
        "below": 6,
        "blue": 7,
        "circle": 8,
        "green": 9,
        "is": 10,
        "left": 11,
        "of": 12,
        "red": 13,
        "right": 14,
        "shape": 15,
        "square": 16,
        "triangle": 17,
    },
    "program_token_to_idx": {
        "<NULL>": 0,
        "<START>": 1,
        "<END>": 2,
        "<UNK>": 3,
        "_And": 4,
        "_Answer": 5,
        "_Find[blue]": 6,
        "_Find[circle]": 7,
        "_Find[green]": 8,
        "_Find[red]": 9,
        "_Find[square]": 10,
        "_Find[triangle]": 11,
        "_Transform[above]": 12,
        "_Transform[below]": 13,
        "_Transform[left_of]": 14,
        "_Transform[right_of]": 15,
    },
    "answer_token_to_idx": {
        "<NULL>": 0,
        "<START>": 1,
        "<END>": 2,
        "<UNK>": 3,
        "False": 4,
        "True": 5,
    },
    "program_token_arity": {
        "<NULL>": 1,
        "<START>": 1,
        "<END>": 1,
        "<UNK>": 1,
        "_And": 2,
        "_Answer": 1,
        "_Find[blue]": 0,
        "_Find[circle]": 0,
        "_Find[green]": 0,
        "_Find[red]": 0,
        "_Find[square]": 0,
        "_Find[triangle]": 0,
        "_Transform[above]": 1,
        "_Transform[below]": 1,
        "_Transform[left_of]": 1,
        "_Transform[right_of]": 1,
    },
}

EXAMPLE_VOCAB["program_idx_to_token"] = {
    EXAMPLE_VOCAB["program_token_to_idx"][key]: key
    for key in EXAMPLE_VOCAB["program_token_to_idx"]
}


def encode_progr(x: List[str]) -> List[int]:
    return (
        [EXAMPLE_VOCAB["program_token_to_idx"]["<START>"]]
        + [EXAMPLE_VOCAB["program_token_to_idx"][token] for token in x]
        + [EXAMPLE_VOCAB["program_token_to_idx"]["<END>"]]
    )


@pytest.fixture()
def symb_shapes_exec():
    vocab = copy.deepcopy(EXAMPLE_VOCAB)
    yield SymbolicShapesExecutor(ClosureVocab(vocab))


def test_immediate_spatial(symb_shapes_exec):
    """
    SHAPES interprets left/right/above/below as immediate operators
    """

    o1 = {
        "id": "0-0",
        "position": [0, 0, 0],
        "color": "red",
        "material": "rubber",
        "shape": "cube",
        "size": "small",
    }
    o2 = {
        "id": "0-1",
        "position": [0, 1, 0],
        "color": "green",
        "material": "rubber",
        "shape": "cube",
        "size": "small",
    }
    o3 = {
        "id": "0-2",
        "position": [1, 0, 0],
        "color": "green",
        "material": "rubber",
        "shape": "cube",
        "size": "small",
    }

    programs = [
        encode_progr(
            ["_Answer", "_And", "_Find[red]", "_Transform[left_of]", "_Find[green]"]
        ),
        encode_progr(
            ["_Answer", "_And", "_Find[green]", "_Transform[left_of]", "_Find[red]"]
        ),
        encode_progr(
            ["_Answer", "_And", "_Find[red]", "_Transform[right_of]", "_Find[green]"]
        ),
        encode_progr(
            ["_Answer", "_And", "_Find[green]", "_Transform[right_of]", "_Find[red]"]
        ),
        encode_progr(
            ["_Answer", "_And", "_Find[red]", "_Transform[above]", "_Find[green]"]
        ),
        encode_progr(
            ["_Answer", "_And", "_Find[green]", "_Transform[above]", "_Find[red]"]
        ),
        encode_progr(
            ["_Answer", "_And", "_Find[red]", "_Transform[below]", "_Find[green]"]
        ),
        encode_progr(
            ["_Answer", "_And", "_Find[green]", "_Transform[below]", "_Find[red]"]
        ),
    ]
    scene_graphs = [copy.deepcopy([o1, o3])] * 8

    # 4 is False, 5 is True
    answers = [5, 4, 4, 5, 4, 4, 4, 4]

    answers = torch.LongTensor(answers)
    programs = torch.LongTensor(programs)
    assert torch.all(symb_shapes_exec(scene_graphs, programs) == answers)
    # Check left/right & different row/column doesn't count
    assert torch.all(symb_shapes_exec([copy.deepcopy([o2, o3])] * 8, programs) == 4)

    scene_graphs = [copy.deepcopy([o1, o2])] * 8
    answers = [4, 4, 4, 4, 5, 4, 4, 5]
    answers = torch.LongTensor(answers)
    assert torch.all(symb_shapes_exec(scene_graphs, programs) == answers)

    # Check it has to be _immediate_ left/right above/below
    o2["position"] = [0, 2, 0]
    o3["position"] = [2, 0, 0]
    assert torch.all(symb_shapes_exec([copy.deepcopy([o1, o3])] * 8, programs) == 4)
    assert torch.all(symb_shapes_exec([copy.deepcopy([o1, o2])] * 8, programs) == 4)
