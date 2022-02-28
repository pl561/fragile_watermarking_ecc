from __future__ import print_function
import sys
import os
HOME = os.environ["HOME"]

import numpy as np
import pandas as pd

import arrayprocessing as ap
from qim import QIM
from dct import dct2d_byblock, idct2d_byblock
from attacks import jpeg_compression
from pathtools import EPath
import metrics as mtc


def bench_binarymap():
    ep_lena = EPath(HOME).join("Images/lena.png")
    print(ep_lena)
    img_x = ep_lena.imread_gs_int()
    msg = np.random.choice([0, 1], img_x.shape)
    w, h = img_x.shape
    nb_pixel = float(w*h)

    delta = 50
    qim = QIM(delta)

    dct_lena = dct2d_byblock(img_x)
    dct_lena_y = qim.embed2(dct_lena, msg)
    img_y = idct2d_byblock(dct_lena_y)

    img_z = jpeg_compression(img_y, 50)

    img_z_dct = dct2d_byblock(img_z)
    img_detected, msg_detected = qim.detect2(img_z_dct)

    bm = mtc.binarymap(msg, msg_detected)
    ber_value = len(np.where(msg != msg_detected)[0])/nb_pixel

    print("PSNR =", mtc.compute_psnr(img_x, img_y))
    print("BER    =", ber_value)
    print("BER LF =", ber_lowfrequency(msg, msg_detected))

    ep_bm = EPath("/tmp").join("qimdct_binarymap.png")
    ep_bm.imwrite(bm)


def ber_lowfrequency(b1, b2):
    ber_value = 0.
    nb_coeff = 0.
    for w0, w1, h0, h1 in ap.sqblock_gen_enum2(b1, 8):
        b1_coeff = b1[w0: w1, h0: h1][0:2,0:2]
        b2_coeff = b2[w0: w1, h0: h1][0:2,0:2]
        nb_coeff += 4
        ber_value += len(np.where(b1_coeff != b2_coeff)[0])
    ber_value /= nb_coeff
    return ber_value


def main():
    bench_binarymap()


if __name__ == "__main__":
    sys.exit(main())
