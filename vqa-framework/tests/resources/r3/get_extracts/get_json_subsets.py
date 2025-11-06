"""
First script used to create MINI-CLOSURE.

For each question family, take the 30 questions using the lowest-number images,
split into 3 groups, and artificially make these the baseline/test/val split.

NOTE: need to manually download full closure and dump it in r3 under a folder called "closure".

NOTE: After this thing finishes generating, still need to create bogus vocab file.
"""

import h5py
import json
import os
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    full_closure_dir = os.path.join("..", "closure")

    for filename in os.listdir(full_closure_dir):
        if "val" in filename and filename.endswith(".json"):
            with open(os.path.join(full_closure_dir, filename), "r") as infile:
                questions_full = json.load(infile)

            questions = list(questions_full["questions"])
            questions.sort(key=lambda x: x["image_index"])
            questions = [
                q for q in questions if q["image_index"] <= 32
            ]  # Some of the .json's skip some images
            for q in questions:
                assert (
                    q["image_filename"]
                    == f"CLEVR_val_{str(q['image_index']).zfill(6)}.png"
                )

            for i, split in enumerate(["val", "test", "baseline"]):
                new_data = {}
                with open(
                    os.path.join(
                        "..", "ori_closure", "closure", filename.replace("val", split)
                    ),
                    "w",
                ) as outfile:
                    new_data["info"] = questions_full["info"]
                    new_data["questions"] = questions[
                        i * (len(questions) // 3) : (i + 1) * (len(questions) // 3)
                    ]
                    json.dump(new_data, outfile, indent=2)
                    print(filename)
                    print(split)
                    print(len(new_data["questions"]))
