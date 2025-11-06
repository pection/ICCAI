#!/usr/bin/env python3

# Modified from CLOSURE code

# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

import json


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, "r") as f:
        vocab = json.load(f)
        vocab["question_idx_to_token"] = invert_dict(vocab["question_token_to_idx"])
        vocab["program_idx_to_token"] = invert_dict(vocab["program_token_to_idx"])
        vocab["answer_idx_to_token"] = invert_dict(vocab["answer_token_to_idx"])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab["question_token_to_idx"]["<NULL>"] == 0
    assert vocab["question_token_to_idx"]["<START>"] == 1
    assert vocab["question_token_to_idx"]["<END>"] == 2
    assert vocab["program_token_to_idx"]["<NULL>"] == 0
    assert vocab["program_token_to_idx"]["<START>"] == 1
    assert vocab["program_token_to_idx"]["<END>"] == 2
    return vocab
