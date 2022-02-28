"""Implementation of DCT using scipy"""


from __future__ import print_function
import os
import time
HOME = os.environ["HOME"]
import sys
import numpy as np
import pandas as pd
import cv2
from scipy.fftpack import dct, idct

import arrayprocessing as ap
from qim import QIM
from pathtools import EPath


def dct2d(x):
    tmp = dct(x, type=2 ,norm='ortho').transpose()
    return dct(tmp, type=2 ,norm='ortho').transpose()


def idct2d(x):
    tmp = idct(x.transpose(), type=2 ,norm='ortho').transpose()
    return idct(tmp, type=2 ,norm='ortho')


def dct2d_byblock(array, blocksize=8):
    """implements dct on square blocks of shape blocksize x blocksize"""
    return ap.process_byblock(array, dct2d, blocksize=blocksize)


def idct2d_byblock(array, blocksize=8):
    """implements idct on square blocks of shape blocksize x blocksize"""
    return ap.process_byblock(array, idct2d, blocksize=blocksize)


def test_dct2d():
    h, w = 512, 512

    while True:
        img = np.random.randint(0, 256, (h, w))

        result = idct2d(dct2d(img))

        cmp = np.allclose(img, result)

        print(cmp)
        assert cmp


def test_dct2d_byblock():
    h, w = 512, 512

    while True:
        img = np.random.randint(0, 256, (h, w))

        result = idct2d_byblock(dct2d_byblock(img))

        cmp = np.allclose(img, result)

        print(cmp)
        assert cmp




def test_dctquantization():
    ep_lena = EPath(HOME).join("Images/lena.png")
    print(ep_lena)
    img_lena = ep_lena.imread_gs_int()
    img_modified = np.zeros_like(img_lena)
    # img_lena = np.random.randint(0, 256, img_lena.shape)
    dct_lena = dct2d_byblock(img_lena)

    qim = QIM(100)
    u, v = 0, 2

    for w0, w1, h0, h1 in ap.sqblock_gen_enum2(img_lena, 8):
        b_sp = img_lena[w0: w1, h0: h1]
        b_dct = dct_lena[w0: w1, h0: h1]

        # print(b_dct)

        dct_modified_2x2 = qim.embed2(b_dct[u:v, u:v], np.ones((v-u, v-u)))
        dct_modified = np.copy(b_dct)
        dct_modified[u:v, u:v] = dct_modified_2x2
        # print(dct_modified)
        # print()
        img_modified[w0: w1, h0: h1] = idct2d(dct_modified)

        # print(b_dct)
        # res = dct2d(idct2d(b_dct).astype(int).astype(float))
        # print(res)
        # print(b_sp)
        # b_sp2 = idct2d(b_dct).astype(int)
        # print(b_sp2)
        # print(np.abs(b_sp2-b_sp))
        # time.sleep(1)

    ep_lena.add_after_stem("modified").imwrite(img_modified)

        # break

        # if b_dct[7, 7] > 100:
        #     print(b_sp)
        #     print(b_dct)
        #     break



def main():
    np.set_printoptions(precision=3, suppress=True)
    test_dctquantization()
    # test_dct2d_byblock()


if __name__ == "__main__":
    sys.exit(main())



