"""
IEP code used by CLOSURE.
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
# workaround, get functions out of old version of scipy
from vqa_framework.data_modules.clevr_scripts.deprecated_scipy import imread, imresize

import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--input_image_dir", required=True)
parser.add_argument("--max_images", default=None, type=int)
parser.add_argument("--output_h5_file", required=True)

parser.add_argument("--image_height", default=224, type=int)
parser.add_argument("--image_width", default=224, type=int)

# Default features are conv4 of ResNet101, should be 1024x14x14
# Matches claim in Tensor-NMN paper https://arxiv.org/pdf/1705.03633.pdf
# Image is rescaled to 224x224 before processing, as it should be
# Image is divided by 255 (i.e., rescaled to 0-1), then z-normalized
# NOTE: colour channels are *not* being transposed before processing with ResNet,
#       they're just re-arranging channels, H, W
parser.add_argument("--model", default="resnet101")
parser.add_argument("--model_stage", default=3, type=int)
parser.add_argument("--batch_size", default=128, type=int)

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


def build_model(args):
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not "resnet" in args.model:
        raise ValueError("Feature extraction only supports ResNets")
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(args.model_stage):
        name = "layer%d" % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def run_batch(cur_batch, model):
    with torch.no_grad():
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

        image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
        image_batch = (image_batch / 255.0 - mean) / std
        image_batch = torch.FloatTensor(image_batch)
        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
        # NOTE: obsolete syntax, replaced with "with torch.no_grad()"
        # image_batch = torch.autograd.Variable(image_batch, volatile=True)

        feats = model(image_batch)
        feats = feats.data.cpu().clone().numpy()

        return feats


def main(args):
    if not hasattr(args, "start_idx"):
        args.start_idx = 0

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

    model = build_model(args)

    img_size = (args.image_height, args.image_width)
    with h5py.File(args.output_h5_file, "w") as f:
        feat_dset = None
        i0 = args.start_idx  # 0
        cur_batch = []
        for i, (path, idx) in enumerate(input_paths):
            img = imread(path, mode="RGB")
            img = imresize(img, img_size, interp="bicubic")
            img = img.transpose(2, 0, 1)[None]
            cur_batch.append(img)
            if len(cur_batch) == args.batch_size:
                feats = run_batch(cur_batch, model)
                if feat_dset is None:
                    N = len(input_paths)
                    if hasattr(args, "N") and args.N is not None:
                        N = args.N
                    _, C, H, W = feats.shape
                    feat_dset = f.create_dataset(
                        "features", (N, C, H, W), dtype=np.float32
                    )
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                i0 = i1
                print(
                    "Processed %d / %d images" % (i1 - args.start_idx, len(input_paths))
                )
                cur_batch = []
        if len(cur_batch) > 0:
            feats = run_batch(cur_batch, model)
            i1 = i0 + len(cur_batch)

            if (
                feat_dset is None
            ):  # added in here due to very small CLEVR testing dataset
                N = len(input_paths)
                if hasattr(args, "N") and args.N is not None:
                    N = args.N
                _, C, H, W = feats.shape
                feat_dset = f.create_dataset("features", (N, C, H, W), dtype=np.float32)

            feat_dset[i0:i1] = feats
            print("Processed %d / %d images" % (i1 - args.start_idx, len(input_paths)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
