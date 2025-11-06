"""
Modified version of ./iep/extract_features.py that puts the SHAPES dataset
into the correct dataset format. Following N2NMN code, we combine all
training sets into one (large, med, small, tiny)

Rearrange into NxCxHxW

*** Very important note! Need to re-order data for colour channels to be
displayed correctly!!! plt.imshow(data[i][:,:,[2, 1, 0]])


Out of paranoia, will divide images by 255 to convert to floats

Because it's unclear what preprocessing is done on the SHAPES dataset,
and it varies between N2NMN and IL, this code only puts the data into
the hdf5 format. Further processing on the user.

In the Iterated Learning paper, only processing done is z-normalization.

NOTE: IL claims image standardization; doesn't state if image-wise or dataset-wise
N2NMN uses dataset-wise, but only subtracts mean. Mean computed over
train.*.input.npy files, but summed over all channels instead of keeping
separate.
"""

import argparse, os
import h5py
import numpy as np
import logging
from argparse import Namespace
from typing import List

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


def process_shapes(name: str, component_files: List[str], args: Namespace):
    target_path = os.path.join(args.output_dir, f"{name}_features.h5")
    if os.path.exists(target_path):
        logging.warning(f"SKIPPING {target_path}, exists")
        return

    with h5py.File(target_path, "w") as f:
        limit_remaining = args.max_images
        curr_batch = []

        for component in component_files:
            if limit_remaining is not None and limit_remaining == 0:
                break

            data = np.load(os.path.join(args.input_dir, f"{component}.input.npy"))
            data = data.transpose((0, 3, 1, 2))
            assert data.shape == (len(data), 3, 30, 30)
            if limit_remaining is not None:
                curr_batch.append(data[:limit_remaining])
                limit_remaining -= len(curr_batch[-1])
            else:
                curr_batch.append(data)

        curr_batch = np.concatenate(curr_batch)
        curr_batch = curr_batch / 255
        N, C, H, W = curr_batch.shape
        feat_dset = f.create_dataset("features", (N, C, H, W), dtype=np.float32)
        feat_dset[:] = curr_batch


def main(args):
    for name, component_files in [
        ("train", ["train.large", "train.med", "train.small", "train.tiny"]),
        ("val", ["val"]),
        ("test", ["test"]),
    ]:
        process_shapes(name=name, component_files=component_files, args=args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
