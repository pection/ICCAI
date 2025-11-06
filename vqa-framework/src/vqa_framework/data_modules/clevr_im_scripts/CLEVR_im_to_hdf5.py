"""
Modified version of ./iep/extract_features.py that puts the CLEVR dataset
images instead of features into the hdf5. Hopefully.

Rearrange into NxCxHxW

*** Very important note! Need to re-order data for colour channels to be
displayed correctly!!! plt.imshow(data[i][:,:,[2, 1, 0]])


Out of paranoia, will divide images by 255 to convert to floats

This code only puts the image data into
the hdf5 format. Further processing on the user.

Modification of IEP code used by CLOSURE.
"""

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os

import h5py
import numpy as np

# from scipy.misc import (
#    imread,
#    imresize,
# )
# Workaround, get functions out of old version of scipy
from vqa_framework.data_modules.clevr_scripts.deprecated_scipy import imread

parser = argparse.ArgumentParser()
parser.add_argument("--input_image_dir", required=True)
parser.add_argument("--max_images", default=None, type=int)
parser.add_argument("--output_h5_file", required=True)
# This argument that's only really useful if you're planning on
# adding more images later:
parser.add_argument(
    "--N",
    default=None,
    type=int,
    help="Specify the number of elements to include in the created h5 file. If not specified, make it equal to the number of images.",
)
parser.add_argument(
    "--start_idx",
    default=0,
    type=int,
    help="Specify the index at which to start inserting images; virtually always going to be 0.",
)

# NOTE: Absolutely no processing.

# NOTE: W.r.t. processing, keep the division by 255, otherwise ditch the preprocessing


def main(args):
    if not hasattr(args, "start_idx"):
        args.start_idx = 0

    args.batch_size = 128

    input_paths = []
    idx_set = set()
    for fn in os.listdir(args.input_image_dir):
        if not fn.endswith(".png"):
            continue
        idx = int(os.path.splitext(fn)[0].split("_")[-1])
        input_paths.append((os.path.join(args.input_image_dir, fn), idx))
        idx_set.add(idx)
    input_paths.sort(key=lambda x: x[1])

    assert len(idx_set) == len(input_paths)
    if not hasattr(args, "N") or args.N is None:
        assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
    else:
        assert min(idx_set) >= 0 and max(idx_set) < args.N
        # Check that the indices are contiguous
        idx_set_list = sorted(list(idx_set))
        for i in range(len(idx_set_list)):
            assert idx_set_list[i] == idx_set_list[0] + i
        del idx_set_list

    if args.max_images is not None:
        input_paths = input_paths[: args.max_images]
    print(input_paths[0])
    print(input_paths[-1])

    with h5py.File(args.output_h5_file, "w") as f:
        feat_dset = None
        i0 = args.start_idx  # 0
        cur_batch = []
        for i, (path, idx) in enumerate(input_paths):
            img = imread(path, mode="RGB")  # HxWxC, with RGB colour channel order
            img = img.transpose(2, 0, 1)  # CxHxW, with RGB colour channel order
            cur_batch.append(img)
            if len(cur_batch) == args.batch_size:
                feats = np.stack(cur_batch, 0)  # run_batch(cur_batch, model)
                if feat_dset is None:
                    N = len(input_paths)
                    if hasattr(args, "N") and args.N is not None:
                        N = args.N
                    _, C, H, W = feats.shape
                    feat_dset = f.create_dataset(
                        "features", (N, C, H, W), dtype=np.ubyte
                    )
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                i0 = i1
                print(
                    "Processed %d / %d images" % (i1 - args.start_idx, len(input_paths))
                )
                cur_batch = []
        if len(cur_batch) > 0:
            feats = np.stack(cur_batch, 0)  # run_batch(cur_batch, model)
            i1 = i0 + len(cur_batch)

            if (
                feat_dset is None
            ):  # added in here due to very small CLEVR testing dataset
                N = len(input_paths)
                if hasattr(args, "N") and args.N is not None:
                    N = args.N
                _, C, H, W = feats.shape
                feat_dset = f.create_dataset("features", (N, C, H, W), dtype=np.ubyte)

            feat_dset[i0:i1] = feats
            print("Processed %d / %d images" % (i1 - args.start_idx, len(input_paths)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
