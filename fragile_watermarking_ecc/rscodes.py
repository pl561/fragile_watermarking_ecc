from __future__ import print_function
import os

HOME = os.environ["HOME"]

import sys
import numpy as np
from sage.all import *
from termcolor import colored
import arrayprocessing as ap
from ecc import sqrt_int
from display import print_vec


class ReedSolomon:
    """Implements a special construction of EL codes over GF(2)
    m is the number of blocks, t is the number of bits in a single block"""
    def __init__(self, s, n, k=1, q=2):
        self.q = q
        self.s = s
        self.k = k
        self.n = n
        self.s0 = sqrt_int(self.s)
        self.n0 = sqrt_int(self.n)
        self.field = GF(self.q**self.s, 'a')

        self.msgspace = VectorSpace(GF(self.q), self.k)
        self.vs_n = VectorSpace(self.field, self.n)
        self.vs_ffelt = VectorSpace(self.field.base_ring(), self.s)
        eval_elts = self.field.list()[1:n+1]
        self.rs = codes.GeneralizedReedSolomonCode(eval_elts, self.k)

    def random_msg(self):
        return self.msgspace.random_element()

    def random_codeword(self, return_block=False):
        cw = self.rs.random_element()
        if return_block:
            block = self.to_block(cw)
            return block
        else:
            return cw


    def to_block(self, c):
        shape = (self.s0*self.n0,)*2
        block_sb = np.zeros(shape).astype(int)
        i = 0
        gen = ap.sqblock_gen_enum(block_sb, self.s0*self.n0, self.s0)
        for w0, w1, h0, h1 in gen:
            cw_sb = np.array(tuple(self.vs_ffelt(c[i])))
            block_sb[w0: w1, h0: h1] = cw_sb.reshape(self.s0, self.s0)
            i += 1
        return block_sb


    def to_vec(self, block):
        """transforms a block into its corresponding vector representation
        it is the inverse operation of to_block method"""
        c = []

        gen = ap.sqblock_gen_enum(block, self.s0*self.n0, self.s0)
        for w0, w1, h0, h1 in gen:
            block_sb = block[w0: w1, h0: h1]
            flattened = self.vs_ffelt(tuple(block_sb.flatten()))
            ffelt = self.field(flattened)
            c.append(ffelt)
        c = self.vs_n(tuple(c))
        return c


def test_RS_class():
    # code = ErrorLocatingCodeSp2(4, 4)
    # code.print_allsyndromes()
    code = ReedSolomon(9, 4, k=2)

    # print(code.H)


    for i in range(10000):
        cw = code.random_codeword()

        block = code.to_block(cw)
        print(cw)
        # print_vec("CW  = ", cw, sep1=", ", sep2="  ",
        #           step=4, endline=True)
        cw2 = code.to_vec(block)

        print(cw2)
        # print_vec("CW2 = ", cw2, sep1=", ", sep2="  ",
        #           step=4, endline=True)
        print()
        assert cw == cw2
        print(i)


def main():
    test_RS_class()


if __name__ == "__main__":
    sys.exit(main())