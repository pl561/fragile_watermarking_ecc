from __future__ import print_function
import os

import sys
import numpy as np
from sage.all import *

bch_params = {
    # (s, n) : (n_bch, k_bch, t_bch
    (4, 9) : (31, 6, 7),
    (9, 4) : (31, 6, 7),

    (4, 16)  : (63, 7, 15),
    (16, 4)  : (63, 7, 15),

    (9, 16)  : (127, 8, 31),
    (16, 9)  : (127, 8, 31),

    (16, 16) : (255, 9, 63),

    (9, 64)  : (512, 10, 121),
    (64, 9)  : (512, 10, 121),

    (64, 16)  : (1023, 11, 255),
    (16, 64)  : (1023, 11, 255),
}

bch_codelength = [
    (15, 5, 3),
    (31, 6, 7),
    (63, 7, 15),
    (127, 8, 31),
    (255, 9, 63),
    (511, 10, 127),
    (1023, 11, 255),
    (2047, 12, 511),
]


def generate_bch_params():
    n_values = [2**i - 1 for i in range(4, 12)]
    gf2 = GF(2)
    for n in n_values:
        print()
        for d in range(n-1, 0, -1):
            code = codes.BCHCode(gf2, n, d)
            # print(code)
            k = code.dimension()
            if k != 1:
                t = d//2
                print("n = {} k = {} t = {}".format(n, k, t))
                break


def main():
    generate_bch_params()


if __name__ == "__main__":
    sys.exit(main())
