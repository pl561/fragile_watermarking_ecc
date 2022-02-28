import os
HOME = os.environ["HOME"]

import sys
import glob
import numpy as np
import cv2

def get_tampered_realistic_png(model="Canon", basenames=False):
    path = os.path.join(HOME, "Images/realistic_tampering_dataset")
    if model == "Canon":
        dspath = os.path.join(path, "Canon_60D_png")
        dpp_fnames_x = sorted(glob.glob(dspath+"/*_x.png"))
        dpp_fnames_t = sorted(glob.glob(dspath+"/*_t.png"))
        if basenames:
            dpp_fnames_x = [os.path.basename(f) for f in dpp_fnames_x]
            dpp_fnames_t = [os.path.basename(f) for f in dpp_fnames_t]

        return dpp_fnames_x, dpp_fnames_t
    else:
        raise ValueError("Unknown camera model : {}".format(model))


def get_tampered_realistic_png_resized(model="Canon", basenames=False):
    path = os.path.join(HOME, "Images/realistic_tampering_dataset")
    if model == "Canon":
        dspath = os.path.join(path, "Canon_60D_png_resized")
        dpp_fnames_x = sorted(glob.glob(dspath+"/*_x.png"))
        dpp_fnames_t = sorted(glob.glob(dspath+"/*_t.png"))
        if basenames:
            dpp_fnames_x = [os.path.basename(f) for f in dpp_fnames_x]
            dpp_fnames_t = [os.path.basename(f) for f in dpp_fnames_t]

        return dpp_fnames_x, dpp_fnames_t
    else:
        raise ValueError("Unknown camera model : {}".format(model))


def test_call_dataset():
    fnames_x, fnames_t = get_tampered_realistic_png(basenames=True)
    for fname_x, fname_t in zip(fnames_x, fnames_t):
        print(fname_x, fname_t)

    print(len(fnames_t))


def main():
    test_call_dataset()


if __name__ == "__main__":
    sys.exit(main())