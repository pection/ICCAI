"""
Script to generate scene graphs for SHAPES images

NOTE: puts data into CLEVR scene annotation format; not exactly equiv
(2D vs. 3D); to compensate some fields are set to dummy values (e.g. material),
and we consider further down the image to be "in front"

NOTE: need to alter left/right semantics, I'm pretty SHAPES requires immediate
left/right (i.e., left & same height)

NOTE: maps 'square' -> cube
           'circle' -> sphere
           'triangle' -> cylinder
"""

import argparse
import json
import h5py
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

DEBUG = False


def np_check_in(item, list):
    for item2 in list:
        if np.all(item == item2):
            return True
    return False


def is_square(grid):
    first_non_false = None
    for row in grid:
        if np.any(row):
            first_non_false = row
            break

    assert first_non_false is not None

    cont_len = 0
    start_continuous = False
    end_continuous = False

    for val in first_non_false:
        start_continuous = start_continuous or val
        cont_len += val
        if start_continuous and not val:
            end_continuous = True
        if val and end_continuous:
            return False

    start_continuous = False
    end_continuous = False
    repeats = 0

    for row in grid:
        match = np.all(row == first_non_false)
        repeats += match
        start_continuous = start_continuous or match
        if start_continuous and not match:
            end_continuous = True
        if match and end_continuous:
            return False

    return repeats == cont_len


def block_out(corner: int):
    grid = np.zeros((10, 10), dtype=np.bool)
    for x in range(10):
        for y in range(10):
            if min(x, 9 - x, y, 9 - y) < corner:
                grid[x, y] = True
    return grid


"""
def expand_to_square(grid):
    start_i = 0
    first_non_false = None
    for i, row in enumerate(grid):
        if np.any(row):
            first_non_false = row
            start_i = i
            break
    assert first_non_false is not None


    true_x = [i for i in range(len(first_non_false)) if first_non_false[i]]
    start_true_x = min(true_x)
    end_true_x = max(true_x)
    assert start_true_x > 0
    assert end_true_x < 9
    assert len(true_x) <= 8

    new_row = np.copy(first_non_false)
    new_row[start_true_x - 1] = True
    new_row[end_true_x + 1] = True
    new_grid = np.zeros((10,10), dtype=np.bool)

    for i in range(start_i - 1, start_i - 1 + len(true_x) + 2):
        new_grid[i] = new_row

    return new_grid
"""


def main2(name, args):
    save_path = os.path.join(args.h5_dir, f"{name}_scenes.json")
    if os.path.exists(save_path):
        logging.info(f"SKIP {name}, {save_path} exists")
        return

    with h5py.File(os.path.join(args.h5_dir, f"{name}_features.h5"), "r") as f:
        ims = f["features"]

        result = {
            "info": {
                "split": name,
                "license": "Creative Commons Attribution (CC BY 4.0)",
                "version": "1.0",
                "date": datetime.now().strftime("%m/%d/%Y"),
            },
            "scenes": [],
        }
        # Note directions, bigger x is front (i.e., towards bottom of img)
        #                  bigger y is right

        # logging.info(ims)

        for i in range(ims.shape[0]):
            if DEBUG:
                if i < 23:  # 54: #49: #31:
                    continue

            img = ims[i]
            if args.max_images is not None and i >= args.max_images:
                break

            new_scene = {
                "image_index": i,
                "objects": [],
                "relationships": {"right": [], "behind": [], "front": [], "left": []},
                "image_filename": f"{name}_features.h5",
                "split": name,
                "directions": {
                    "right": [0, 1, 0],
                    "behind": [-1, 0, 0],
                    "above": [0, 0, 1],
                    "below": [0, 0, -1],
                    "left": [0, -1, 0],
                    "front": [1, 0, 0],
                },
            }
            assert img.shape == (3, 30, 30)

            # Re-order colour channels so that they display properly in
            # matplotlib
            img = img[[2, 1, 0], :, :]

            for x in range(3):
                for y in range(3):
                    if DEBUG:
                        logging.info(f"{i}: ({x}, {y})")
                    # 1) check what shape is in this position
                    grid = img[:, y * 10 : (y + 1) * 10, x * 10 : (x + 1) * 10]
                    assert grid.shape == (3, 10, 10)

                    tmp_flat_grid = np.sum(grid, axis=0, keepdims=False)
                    assert tmp_flat_grid.shape == (10, 10)

                    if np.amax(tmp_flat_grid) == 0:
                        continue  # Nothing in this grid square
                    else:
                        new_obj = {
                            "size": "small",
                            "rotation": 0,
                            "3d_coords": [y, x, 0],
                            "material": "rubber",
                            "pixel_coords": [
                                y * 10 + 5,
                                x * 10 + 5,
                                0,  # The last value is not used by the loader, so it is safe to ignore
                            ],
                        }
                        # 1b) find the colour
                        tmp = np.amax(grid, axis=(1, 2))
                        colour = np.argmax(tmp)
                        if colour == 0:
                            if DEBUG:
                                logging.info("RED ")
                            new_obj["color"] = "red"
                        elif colour == 1:
                            if DEBUG:
                                logging.info("GREEN ")
                            new_obj["color"] = "green"
                        elif colour == 2:
                            if DEBUG:
                                logging.info("BLUE ")
                            new_obj["color"] = "blue"
                        else:
                            raise Exception("too many channesl")

                        flat_grid = grid[colour, :, :].copy()
                        assert flat_grid.shape == (10, 10)
                        # flat_grid /= 0.4
                        # flat_grid = np.clip(flat_grid, 0, 1)

                        if np.amax(flat_grid - np.flipud(flat_grid)) > 0.5:
                            new_obj["shape"] = "cylinder"
                            if DEBUG:
                                logging.info("TRIANGLE")
                                continue
                        else:
                            bool_flat_grid = (
                                flat_grid >= np.amax(flat_grid) - 0.001
                            )  # 0.001 works very well
                            corner = None
                            for cy in range(10):
                                for cx in range(10):
                                    if bool_flat_grid[cx, cy]:
                                        corner = [cx, cy]
                                        break
                            assert corner is not None
                            # corner[0] -= 1
                            # assert corner[0] >= 0
                            corner = min(corner)

                            if is_square(bool_flat_grid):
                                if DEBUG:
                                    logging.info("SQUARE")
                                    continue
                                new_obj["shape"] = "cube"
                            elif flat_grid[corner, corner] > 0.95:
                                new_bool_grid = block_out(corner - 1)
                                # logging.info(np.where(new_bool_grid, flat_grid, 0))
                                if (
                                    np.amax(np.where(new_bool_grid, flat_grid, 0))
                                    < 0.01
                                ):
                                    if DEBUG:
                                        logging.info("SQUARE")
                                    new_obj["shape"] = "cube"
                                else:
                                    if DEBUG:
                                        logging.info("CIRCLE")
                                        continue
                                    new_obj["shape"] = "sphere"
                            else:
                                if DEBUG:
                                    logging.info("CIRCLE")
                                    continue
                                new_obj["shape"] = "sphere"

                            if DEBUG:
                                logging.info(corner)
                            """
                            tmp = str(bool_flat_grid)
                            tmp = tmp.replace("True ", "True, ")
                            tmp = tmp.replace("False ", "False, ")
                            tmp = tmp.replace(']\n', '],\n')
                            logging.info(tmp)
                            """
                        if DEBUG:
                            plt.imshow(grid.transpose(1, 2, 0))
                            plt.show()
                        new_scene["objects"].append(new_obj)

            # Populate relationships
            # NOTE: modified for _direct_ left/right up/down
            for key in ["right", "behind", "front", "left"]:
                new_scene["relationships"][key] = [
                    list() for i in range(len(new_scene["objects"]))
                ]
            for i in range(len(new_scene["objects"])):
                for j in range(len(new_scene["objects"])):
                    if (
                        new_scene["objects"][i]["3d_coords"][0]
                        < new_scene["objects"][j]["3d_coords"][0]
                        and new_scene["objects"][i]["3d_coords"][1]
                        == new_scene["objects"][j]["3d_coords"][1]
                    ):
                        new_scene["relationships"]["front"][i].append(j)
                    elif (
                        new_scene["objects"][i]["3d_coords"][0]
                        > new_scene["objects"][j]["3d_coords"][0]
                        and new_scene["objects"][i]["3d_coords"][1]
                        == new_scene["objects"][j]["3d_coords"][1]
                    ):
                        new_scene["relationships"]["behind"][i].append(j)

                    if (
                        new_scene["objects"][i]["3d_coords"][1]
                        > new_scene["objects"][j]["3d_coords"][1]
                        and new_scene["objects"][i]["3d_coords"][0]
                        == new_scene["objects"][j]["3d_coords"][0]
                    ):
                        new_scene["relationships"]["left"][i].append(j)
                    elif (
                        new_scene["objects"][i]["3d_coords"][1]
                        < new_scene["objects"][j]["3d_coords"][1]
                        and new_scene["objects"][i]["3d_coords"][0]
                        == new_scene["objects"][j]["3d_coords"][0]
                    ):
                        new_scene["relationships"]["right"][i].append(j)

            result["scenes"].append(new_scene)

        with open(save_path, "w") as outfile:
            json.dump(result, outfile, indent=2)


def main(args):
    for name in ["val", "train", "test"]:
        main2(name, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_dir",
        required=True,
        help="path to dir with previously generated features h5 files",
    )
    parser.add_argument(
        "--max_images", default=None, type=int, help="Upper limit on images"
    )
    args = parser.parse_args()

    main(args)
