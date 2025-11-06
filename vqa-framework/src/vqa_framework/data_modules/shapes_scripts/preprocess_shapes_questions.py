"""
Modified version of preprocess_questions.py that puts SHAPES into same
dataformat.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(""))

import argparse

import json

import h5py
import numpy as np

from vqa_framework.vr.preprocess import tokenize, encode, build_vocab

import logging
import vqa_framework.vr.programs as vr_programs

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="prefix", choices=["chain", "prefix", "postfix"])
parser.add_argument("--input_questions_json", required=True, action="append")
# NOTE: family shift probably due to format of CLEVR code;
# each family of questions gets own index; CLOSURE generation probably set
# family back to 0, therefore need to shift by number of CLEVR families
parser.add_argument("--input_vocab_json", default="")
parser.add_argument("--expand_vocab", default=0, type=int)
parser.add_argument("--unk_threshold", default=1, type=int)
parser.add_argument("--encode_unk", default=0, type=int)

parser.add_argument("--output_h5_file", required=True)
parser.add_argument("--output_vocab_json", default="")


def program_to_str(program, mode):
    converter = vr_programs.ProgramConverter()
    if mode == "chain":
        if not converter.is_chain(program):
            return None
        return vr_programs.list_to_str(program)
    elif mode == "prefix":
        program_prefix = converter.list_to_prefix(program)
        return vr_programs.list_to_str(program_prefix)
    elif mode == "postfix":
        program_postfix = converter.list_to_postfix(program)
        return vr_programs.list_to_str(program_postfix)
    return None


def main(args):
    if (args.input_vocab_json == "") and (args.output_vocab_json == ""):
        logging.critical("Must give one of --input_vocab_json or --output_vocab_json")
        return

    logging.info(f"Loading data from{args.input_questions_json}")

    questions = []
    for q_file in args.input_questions_json:
        logging.info(q_file)
        with open(q_file, "r") as f:
            more_questions = json.load(f)["questions"]
            questions.extend(more_questions)

    # Either create the vocab or load it from disk
    if args.input_vocab_json == "" or args.expand_vocab == 1:
        logging.info("Building vocab")
        if "answer" in questions[0]:
            answer_token_to_idx = build_vocab(  # NOTE: IAN: vocab is just a dictionary from token to index
                (str(q["answer"]) for q in questions)
            )
        question_token_to_idx = build_vocab(
            (q["question"] for q in questions),
            min_token_count=args.unk_threshold,
            punct_to_keep=[";", ","],
            punct_to_remove=["?", "."],
        )
        all_program_strs = []
        for q in questions:
            if "program" not in q:
                continue
            program_str = program_to_str(q["program"], args.mode)
            # logging.info(q['program'])
            # logging.info(program_str)
            if program_str is not None:
                all_program_strs.append(program_str)
        program_token_to_idx = build_vocab(all_program_strs)
        vocab = {
            "question_token_to_idx": question_token_to_idx,
            "program_token_to_idx": program_token_to_idx,
            "answer_token_to_idx": answer_token_to_idx,
        }

        def arity(name):
            if "_Find" in name:
                return 0
            if name == "_And":
                return 2
            return 1

        vocab["program_token_arity"] = {
            name: arity(name) for name in program_token_to_idx
        }
    if args.input_vocab_json != "":
        logging.info("Loading vocab")
        if args.expand_vocab == 1:
            new_vocab = vocab
        with open(args.input_vocab_json, "r") as f:
            vocab = json.load(f)
        if args.expand_vocab == 1:
            num_new_words = 0
            for word in new_vocab["question_token_to_idx"]:
                if word not in vocab["question_token_to_idx"]:
                    logging.info("Found new word %s" % word)
                    idx = len(vocab["question_token_to_idx"])
                    vocab["question_token_to_idx"][word] = idx
                    num_new_words += 1
            logging.info("Found %d new words" % num_new_words)

    if args.output_vocab_json != "":
        with open(args.output_vocab_json, "w") as f:
            json.dump(vocab, f)

    # Encode all questions and programs
    logging.info("Encoding data")
    questions_encoded = []
    programs_encoded = []
    question_families = []
    orig_idxs = []
    image_idxs = []
    answers = []
    types = []

    # Record question and program lengths
    questions_len = []
    programs_len = []

    for orig_idx, q in enumerate(questions):
        question = q["question"]
        if "program" in q:
            types += [q["program"][-1]["function"]]

        orig_idxs.append(orig_idx)
        image_idxs.append(q["image_index"])
        if "question_family_index" in q:
            question_families.append(q["question_family_index"])
        question_tokens = tokenize(
            question, punct_to_keep=[";", ","], punct_to_remove=["?", "."]
        )
        question_encoded = encode(
            question_tokens,
            vocab["question_token_to_idx"],
            allow_unk=args.encode_unk == 1,
        )
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))

        if "program" in q:
            program = q["program"]
            program_str = program_to_str(program, args.mode)
            program_tokens = tokenize(program_str)
            program_encoded = encode(program_tokens, vocab["program_token_to_idx"])
            programs_encoded.append(program_encoded)
            programs_len.append(len(program_encoded))

        if "answer" in q:
            answers.append(vocab["answer_token_to_idx"][str(q["answer"])])

    # Pad encoded questions and programs
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab["question_token_to_idx"]["<NULL>"])

    if len(programs_encoded) > 0:
        max_program_length = max(len(x) for x in programs_encoded)
        for pe in programs_encoded:
            while len(pe) < max_program_length:
                pe.append(vocab["program_token_to_idx"]["<NULL>"])

    # Create h5 file
    logging.info("Writing output")
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
    logging.info(questions_encoded.shape)
    logging.info(programs_encoded.shape)

    mapping = {}
    for i, t in enumerate(set(types)):
        mapping[t] = i

    logging.info(mapping)

    types_coded = []
    for t in types:
        types_coded += [mapping[t]]

    with h5py.File(args.output_h5_file, "w") as f:
        f.create_dataset("questions", data=questions_encoded)
        f.create_dataset("image_idxs", data=np.asarray(image_idxs))
        f.create_dataset("orig_idxs", data=np.asarray(orig_idxs))
        f.create_dataset(
            "questions_len", data=np.asarray(questions_len, dtype=np.int32)
        )

        if len(programs_encoded) > 0:
            f.create_dataset("programs", data=programs_encoded)
            f.create_dataset(
                "programs_len", data=np.asarray(programs_len, dtype=np.int32)
            )
        if len(question_families) > 0:
            f.create_dataset("question_families", data=np.asarray(question_families))
        if len(answers) > 0:
            f.create_dataset("answers", data=np.asarray(answers))
        if len(types) > 0:
            f.create_dataset("types", data=np.asarray(types_coded))

    logging.info(json.dumps(vocab, indent=2))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
