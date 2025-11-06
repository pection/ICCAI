"""
Script to convert SHAPES questions into the .json format used by CLEVR.

Mostly parsing the question into the proper structured form, with arguments
(N2NMN has a program parser, but it doesn't parse program arguments)
"""

from datetime import datetime
from typing import List, Dict, Union
import sexpdata
import argparse
import os
import json
import logging

QUESTION_FAMILY = {
    tuple(["_Find", "_Find", "_Transform", "_And", "_Answer"]): 0,
    tuple(["_Find", "_Find", "_Transform", "_Transform", "_And", "_Answer"]): 1,
    tuple(["_Find", "_Find", "_And", "_Answer"]): 2,
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


def main2(name, component_files, in_dir, out_dir, max_ims):
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
            reduced_progr = [x["function"] for x in question_repr["program"]]
            question_repr["question_family_index"] = QUESTION_FAMILY[
                tuple(reduced_progr)
            ]
            question_repr["split"] = component
            answer = answer.strip()
            assert answer in ["true", "false"]
            question_repr["answer"] = answer == "true"
            question_repr["question"] = question.strip()
            result["questions"].append(question_repr)

    with open(os.path.join(out_dir, f"SHAPES_{name}_questions.json"), "w") as outfile:
        json.dump(result, outfile, indent=2)


def main(args):
    for name, component_files in [
        ("train", ["train.large", "train.med", "train.small", "train.tiny"]),
        ("val", ["val"]),
        ("test", ["test"]),
    ]:
        main2(name, component_files, args.input_dir, args.output_dir, args.max_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="path to exp_shapes/shapes_dataset from N2NMN repo",
    )
    parser.add_argument(
        "--max_images", default=None, type=int, help="Upper limit on images"
    )
    parser.add_argument("--output_dir", required=True, help="Save output location")
    args = parser.parse_args()
    main(args)
