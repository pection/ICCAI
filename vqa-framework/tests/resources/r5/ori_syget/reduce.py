"""
Quick program. Put the original CLOSURE files here. Run this. 
It crops them down to 50 questions per family. 
"""

import numpy as np
from collections import defaultdict

COLOURS = ["red", "green", "blue"]
SHAPES = ["triangle", "circle", "square"]
DIR = ["above", "below", "left_of", "right_of"]


def genericize(program: str):
    for col in COLOURS:
        program = program.replace(col, "colour")
    for shp in SHAPES:
        program = program.replace(shp, "shape")
    for direction in DIR:
        program = program.replace(direction, "dir")

    return program


def zero():
    return 0


def load_data(filename: str):
    with open(filename, "r") as infile:
        return infile.readlines()


def save_data(filename: str, data):
    with open(filename, "w") as outfile:
        for dat in data[:-1]:
            print(dat.strip(), file=outfile)
        print(data[-1].strip(), end="", file=outfile)


def index_w_list(data, indices):
    return [dat for i, dat in enumerate(data) if i in indices]


def main(npy_file, out_file, q_file, qs_file):
    observed_progs = defaultdict(zero)

    out_dat = load_data(out_file)
    q_dat = load_data(q_file)
    qs_dat = load_data(qs_file)

    images = np.load(npy_file)

    indices = []

    for i, prog in enumerate(q_dat):
        prog = genericize(prog)
        observed_progs[prog] += 1
        if observed_progs[prog] <= 50:
            indices.append(i)

    out_dat = index_w_list(out_dat, indices)
    q_dat = index_w_list(q_dat, indices)
    qs_dat = index_w_list(qs_dat, indices)

    np.save(npy_file, images[indices, :, :, :])
    save_data(out_file, out_dat)
    save_data(q_file, q_dat)
    save_data(qs_file, qs_dat)


if __name__ == "__main__":
    for x, y, z, w in [
        ("train.input.npy", "train.output", "train.query", "train.query_str.txt"),
        (
            "val_iid.input.npy",
            "val_iid.output",
            "val_iid.query",
            "val_iid.query_str.txt",
        ),
        (
            "val_ood.input.npy",
            "val_ood.output",
            "val_ood.query",
            "val_ood.query_str.txt",
        ),
    ]:
        main(x, y, z, w)
