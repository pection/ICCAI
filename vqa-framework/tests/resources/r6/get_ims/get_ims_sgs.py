"""
Quick & ugly script; given the question .json files from r3, we retrieve the
corresponding images, and split up the scene graphs .json.

Note that we're still not done b/c the indices aren't continuous
"""

import os
import shutil
import json
import copy

FULL_CLEVR_PATH = (
    "/media/administrator/extdrive/vqa-frame/ori_clevr/CLEVR_v1.0/images/val"
)


def get_split(filename: str):
    if "_baseline.json" in filename:
        return "train"
    elif "_val.json" in filename:
        return "val"
    else:
        assert "_test.json" in filename
        return "test"


if __name__ == "__main__":
    # Find the required set of Validation CLEVR images
    required = {"train": set(), "val": set(), "test": set()}
    ori_closure_dir = os.path.join("ori_questions")

    os.makedirs(os.path.join("..", "sgs"))
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join("..", "ims", split))
        os.makedirs(os.path.join("..", "questions", split))

    for filename in os.listdir(ori_closure_dir):
        if filename != "vocab.json" and filename.endswith(".json"):
            split = get_split(filename)

            with open(os.path.join(ori_closure_dir, filename), "r") as infile:
                questions = json.load(infile)

            questions = questions["questions"]
            for q in questions:
                assert (
                    q["image_filename"]
                    == f"CLEVR_val_{str(q['image_index']).zfill(6)}.png"
                )
                required[split].add(q["image_index"])

    with open("CLEVR_val_scenes.json", "r") as infile:
        all_scenes = json.load(infile)

    for split in ["train", "val", "test"]:
        split_scenes = copy.deepcopy(all_scenes)
        split_scenes["scenes"] = [
            x for x in split_scenes["scenes"] if x["image_index"] in required[split]
        ]
        with open(os.path.join("..", "sgs", f"{split}_scenes.json"), "w") as outfile:
            json.dump(split_scenes, outfile, indent=2)

        for image_index in required[split]:
            source = os.path.join(
                FULL_CLEVR_PATH, f"CLEVR_val_{str(image_index).zfill(6)}.png"
            )
            target = os.path.join(
                "..", "ims", split, f"CLEVR_val_{str(image_index).zfill(6)}.png"
            )
            shutil.copy(source, target)
