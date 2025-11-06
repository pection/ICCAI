"""
Convert SyGeT-SHAPES questions into CLEVR style .json file.

Main difference between this and shapes_q_to_json is the question family
computation. SyGeT question families require more details; need to distinguish
colour & shape.
"""

from datetime import datetime
from typing import List, Dict, Union
import sexpdata
import argparse
import os
import json
import logging


QUESTION_FAMILY = {
    # SyGeT Families; numbers from https://github.com/ankitkv/SHAPES-SyGeT
    # Train families
    tuple(["_Find_colour", "_Find_colour", "_Transform", "_And", "_Answer"]): 1,
    tuple(["_Find_shape", "_Find_colour", "_Transform", "_And", "_Answer"]): 2,
    tuple(
        ["_Find_shape", "_Find_colour", "_Transform", "_Transform", "_And", "_Answer"]
    ): 3,
    tuple(
        ["_Find_shape", "_Find_shape", "_Transform", "_Transform", "_And", "_Answer"]
    ): 4,
    tuple(["_Find_colour", "_Find_shape", "_And", "_Answer"]): 5,
    tuple(["_Find_shape", "_Find_colour", "_And", "_Answer"]): 6,
    tuple(["_Find_shape", "_Find_shape", "_And", "_Answer"]): 7,
    # Val OOD families
    tuple(
        ["_Find_colour", "_Find_colour", "_Transform", "_Transform", "_And", "_Answer"]
    ): 8,
    tuple(["_Find_colour", "_Find_shape", "_Transform", "_And", "_Answer"]): 9,
    tuple(
        ["_Find_colour", "_Find_shape", "_Transform", "_Transform", "_And", "_Answer"]
    ): 10,
    tuple(["_Find_shape", "_Find_shape", "_Transform", "_And", "_Answer"]): 11,
    tuple(["_Find_colour", "_Find_colour", "_And", "_Answer"]): 12,
}


def extract_parse(p):
    if isinstance(p, sexpdata.Symbol):
        return p.value()
    elif isinstance(p, int):
        return str(p)
    elif isinstance(p, bool):
        return str(p).lower()
    elif isinstance(p, float):
        return str(p).lower()
    return tuple(extract_parse(q) for q in p)


def parse_tree(p: str):
    if "'" in p:
        logging.warning(f"WARNING: Skipping... something? {p}")
        p = "none"
    parsed = sexpdata.loads(p)
    extracted = extract_parse(parsed)
    return extracted


COLOURS = ["red", "green", "blue"]
SHAPES = ["triangle", "circle", "square"]


def layout_from_parsing(parse):
    if isinstance(parse, str):
        assert parse in COLOURS + SHAPES
        return (["_Find", [parse]],)
    head = parse[0]
    if len(parse) > 2:  # fuse multiple tokens with "_And"
        assert (len(parse)) == 3
        below = (
            ["_And", []],
            layout_from_parsing(parse[1]),
            layout_from_parsing(parse[2]),
        )
    else:
        below = layout_from_parsing(parse[1])
    if head == "is":
        module = ["_Answer", []]
    elif head in ["above", "below", "left_of", "right_of"]:
        module = ["_Transform", [head]]
    return (module, below)


def flatten_layout(module_layout):
    # Postorder traversal to generate Reverse Polish Notation (RPN)
    RPN = []
    module = {"function": module_layout[0][0], "value_inputs": module_layout[0][1]}
    for m in module_layout[1:]:
        RPN += flatten_layout(m)
    RPN += [module]
    return RPN


def label_inputs(rpn):
    # given RPN, add indexing for the inputs
    stack = []

    for i, instruction in enumerate(rpn):
        if instruction["function"] == "_Find":
            instruction["inputs"] = []
        elif instruction["function"] == "_Transform":
            assert len(stack) >= 1
            instruction["inputs"] = [stack.pop()]
        elif instruction["function"] == "_Answer":
            assert len(stack) == 1
            assert i == len(rpn) - 1
            instruction["inputs"] = [stack.pop()]
        elif instruction["function"] == "_And":
            assert len(stack) >= 2
            instruction["inputs"] = [stack.pop(), stack.pop()]
        else:
            raise NotImplementedError(f"Instruction: {instruction}")

        stack.append(i)
    return rpn


def parse_shapes(instruction: str) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Modified from N2NMN repo (get_ground_truth_layout.ipynb)
    -- that repo only gets function types, here it's
    modified to get function+value

    Processing .query files
    """
    return label_inputs(flatten_layout(layout_from_parsing(parse_tree(instruction))))


def main2(name, component_files, in_dir, out_dir, max_ims, dset_name="SHAPES"):
    result = {
        "info": {
            "split": name,
            "license": "Creative Commons Attribution (CC BY 4.0)",
            "version": "1.0",
            "date": datetime.now().strftime("%m/%d/%Y"),
        },
        "questions": [],
    }

    # image_index
    # program   -   list of
    """
    {
        "inputs": [
            0
        ],
        "function": "filter_size",
        "value_inputs": [
            "large"
        ]
    },
    """
    # question_index
    # image_filename
    # question_family_index
    # split
    # answer
    # question

    for component in component_files:
        with open(os.path.join(in_dir, f"{component}.query_str.txt"), "r") as infile:
            questions = infile.readlines()
        with open(os.path.join(in_dir, f"{component}.query"), "r") as infile:
            programs = infile.readlines()
        with open(os.path.join(in_dir, f"{component}.output"), "r") as infile:
            answers = infile.readlines()

        assert len(questions) == len(programs)
        assert len(questions) == len(answers)

        for i, (question, program, answer) in enumerate(
            zip(questions, programs, answers)
        ):
            if max_ims is not None and len(result["questions"]) >= max_ims:
                break

            question_repr = {}
            question_repr["image_index"] = i
            question_repr["program"] = parse_shapes(program)
            question_repr["question_index"] = i
            question_repr["image_filename"] = f"{component}.input.npy"
            reduced_progr = [prog_atom_name(x) for x in question_repr["program"]]
            question_repr["question_family_index"] = QUESTION_FAMILY[
                tuple(reduced_progr)
            ]
            question_repr["split"] = component
            answer = answer.strip()
            assert answer in ["true", "false"]
            question_repr["answer"] = answer == "true"
            question_repr["question"] = question.strip()
            result["questions"].append(question_repr)

    with open(
        os.path.join(out_dir, f"{dset_name}_{name}_questions.json"), "w"
    ) as outfile:
        json.dump(result, outfile, indent=2)


def prog_atom_name(x):
    if x["function"] != "_Find":
        return x["function"]
    else:
        assert len(x["value_inputs"]) == 1
        if x["value_inputs"][0] in COLOURS:
            arg_name = "colour"
        elif x["value_inputs"][0] in SHAPES:
            arg_name = "shape"
        else:
            raise NotImplementedError(f"Unknown arg: {x['value_inputs'][0]}")

        return x["function"] + "_" + arg_name


if __name__ == "__main__":
    """
    Train templates:

    1. is a COLOR shape TRANSFORM a COLOR shape
    2. is a SHAPE TRANSFORM a COLOR shape
    3. is a SHAPE TRANSFORM TRANSFORM a COLOR shape
    4. is a SHAPE TRANSFORM TRANSFORM a SHAPE
    5. is a COLOR shape a SHAPE
    6. is a SHAPE COLOR
    7. is a SHAPE a SHAPE

    Evaluation templates:

    8. is a COLOR shape TRANSFORM TRANSFORM a COLOR shape
    9. is a COLOR shape TRANSFORM a SHAPE
    10. is a COLOR shape TRANSFORM TRANSFORM a SHAPE
    11. is a SHAPE TRANSFORM a SHAPE
    12. is a COLOR shape COLOR
    """

    for i, tmp in enumerate(
        [
            # val iid
            "(is red (above green))",  # 1
            "(is square (left_of red))",  # 2
            "(is square (right_of (below green)))",  # 3
            "(is circle (above (left_of circle)))",  # 4
            "(is red circle)",  # 5
            "(is triangle blue)",  # 6
            "(is triangle square)",  # 7
            # val ood
            "(is green (left_of (above red)))",  # 8
            "(is green (right_of square))",  # 9
            "(is blue (left_of (left_of triangle)))",  # 10
            "(is circle (right_of square))",  # 11
            "(is green green)",  # 12
        ]
    ):

        # print(parse_shapes(tmp))
        # print()
        print(f"tuple({[prog_atom_name(x) for x in parse_shapes(tmp)]}): {i+1},")
