"""
Looks at which CLEVR validation images are used by CLOSURE, and create a
mini-dataset of just these.

NOTE: You want to *first* create the mini-subset of CLOSURE questions using
get_json_subsets.py

"""

import h5py
import json
import os
from tqdm import tqdm
import numpy as np

FULL_CLEVR_PATH = "/media/administrator/extdrive/vqa-frame/hdf5_clevr/"

if __name__ == "__main__":
    # Find the required set of Validation CLEVR images
    required = set()  # required val image indices
    ori_closure_dir = os.path.join("..", "ori_closure", "closure")

    for filename in os.listdir(ori_closure_dir):
        if filename != "vocab.json" and filename.endswith(".json"):
            with open(os.path.join(ori_closure_dir, filename), "r") as infile:
                questions = json.load(infile)

            questions = questions["questions"]
            for q in questions:
                assert (
                    q["image_filename"]
                    == f"CLEVR_val_{str(q['image_index']).zfill(6)}.png"
                )
                required.add(q["image_index"])

        """
        if "val" in filename:
            required = list(required)
            required.sort()
            print(filename)
            print(required[:30])
        """

    # Create smaller hdf5 CLEVR files with only these images.
    for feat_type in ["ims", "features"]:
        print(feat_type)
        clevr_data = h5py.File(
            os.path.join(FULL_CLEVR_PATH, f"val_{feat_type}.h5"), "r"
        )

        N = 33  # Need the indexing to match. Our json subset code means we're OK.
        if feat_type == "features":
            # clevr_dtype = np.float32
            clevr_dtype = np.ubyte
            C = 1024
            H = 14
            W = 14

        else:
            assert feat_type == "ims"
            clevr_dtype = np.ubyte
            C = 3
            H = 320
            W = 480

        with h5py.File(
            os.path.join("..", "hdf5_clevr", f"val_{feat_type}.h5"), "w"
        ) as f:
            feat_dset = f.create_dataset("features", (N, C, H, W), dtype=clevr_dtype)
            for i in tqdm(required):
                feat_dset[i] = clevr_data["features"][i]
