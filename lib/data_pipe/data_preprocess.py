"""
We encode the blp file into npy, which make the validation data can be read by numpy.
"""
import os
import bcolz
import argparse

import numpy as np

import torch
import torchvision
from torchvision.utils import save_image


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode="r")
    issame = np.load(os.path.join(path, "{}_list.npy".format(name)))
    return carray, issame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for preprocess')
    parser.add_argument(
        "--rec_path",
        help="mxnet record file path",
        default='.',
        type=str)
    parser.add_argument("--data", type=str)
    args = parser.parse_args()

    carray, issame = get_val_pair(args.rec_path, args.data)

    np_carray = np.array(carray)
    np.save(
        os.path.join(
            args.rec_path,
            "{}_data.npy".format(
                args.data)),
        np_carray)
