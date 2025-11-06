"""
After running get_ims_sgs.py, we need to correct the image indices for "val" & "train" so that
they are from 0 to # ims.
"""

import os
import json
import copy


def matching(filename, split):
    if split == "train":
        return "_baseline.json" in filename
    if split == "val":
        return "_val.json" in filename
    if split == "test":
        return "_test.json" in filename
    else:
        raise NotImplementedError


for split in ["train", "test"]:
    index_mapping = {}

    # Create index mapping
    new_index = 0
    for filename in os.listdir(os.path.join("..", "ims", split)):
        assert filename.endswith(".png")
        index = int(filename[10:-4])
        index_mapping[index] = new_index
        new_index += 1

    # Rename image files
    tmp = list(os.listdir(os.path.join("..", "ims", split)))
    for filename in tmp:
        if f"{split}_0" in filename:
            continue

        print(filename)
        index = int(filename[10:-4])
        os.rename(
            os.path.join("..", "ims", split, filename),
            os.path.join(
                "..", "ims", split, f"{split}_{str(index_mapping[index]).zfill(6)}.png"
            ),
        )

    # Correct scene graph files
    with open(os.path.join("..", "sgs", f"{split}_scenes.json"), "r") as infile:
        scenes = json.load(infile)

    fixed_scenes = copy.deepcopy(scenes)

    for fix_scene, old_scene in zip(fixed_scenes["scenes"], scenes["scenes"]):
        fix_scene["image_index"] = index_mapping[old_scene["image_index"]]
        fix_scene["image_filename"] = (
            f"{split}_{str(fix_scene['image_index']).zfill(6)}.png"
        )

    # Sort scenes by image index
    fixed_scenes["scenes"].sort(key=lambda x: x["image_index"])

    # Save corrected scene graph files
    with open(os.path.join("..", "sgs", f"{split}_scenes.json"), "w") as outfile:
        scenes = json.dump(fixed_scenes, outfile, indent=2)

    # Fix & copy the question files:
    for filename in os.listdir("ori_questions"):
        if filename != "vocab.json" and filename.endswith(".json"):
            if not matching(filename, split):
                continue

            with open(os.path.join("ori_questions", filename), "r") as infile:
                questions = json.load(infile)

            fixed_questions = copy.deepcopy(questions)

            for fixed_q, ori_q in zip(
                fixed_questions["questions"], questions["questions"]
            ):
                fixed_q["image_index"] = index_mapping[ori_q["image_index"]]
                fixed_q["image_filename"] = (
                    f"{split}_{str(fixed_q['image_index']).zfill(6)}.png"
                )
                fixed_q["image"] = f"{split}_{str(fixed_q['image_index']).zfill(6)}"

            with open(os.path.join("..", "questions", split, filename), "w") as outfile:
                json.dump(fixed_questions, outfile, indent=2)


# Copy val question files (no changes needed):
for filename in os.listdir("ori_questions"):
    if filename != "vocab.json" and filename.endswith(".json"):
        if not matching(filename, "val"):
            continue

        with open(os.path.join("ori_questions", filename), "r") as infile:
            questions = json.load(infile)

        with open(os.path.join("..", "questions", "val", filename), "w") as outfile:
            json.dump(questions, outfile, indent=2)
