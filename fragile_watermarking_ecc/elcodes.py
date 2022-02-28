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

def repetition_parity_matrix(n):
    """:return a repetition parity matrix of size (n-1)*n"""
    parity_matrix = np.zeros((n - 1, n)).astype(int)
    for i in range(n - 1):
        parity_matrix[i, i] = 1
    parity_matrix[:, -1] = 1
    return parity_matrix


def el_parity_matrix(t, m, r1=1):
    """:return a special construction of EL code parity matrix
    r1 is the number of additional rows"""
    if r1 > 1:
        raise ValueError("you must give parity rows")
    n = m * t
    field = GF(2)

    # r1 = 2
    # r1 = 1
    r = n - m + r1
    ms = MatrixSpace(field, r, n)
    pm = np.zeros((r, n)).astype(int)

    sub_pm = repetition_parity_matrix(t)
    # print(sub_pm)
    for i in range(m):
        # print(pm[i*(t-1):i*(t-1)+t-1, i*t:i*t+t])
        pm[i * (t - 1):i * (t - 1) + t - 1, i * t:i * t + t] = sub_pm

    u = [1] + [0] * (t - 1)
    last_row = u * m
    pm[-1, :] = np.array(last_row)

    # u = [1] + [0] * (t - 1)
    # for j in range(n-m, r-1):
    #     pr =
    #     pm[j, :] = pr
    #
    # pm[r-1, :] =
    # pour ameliorer la matrice de parite,
    # il faudrait modifier l'algo de calcul de syndrome
    # et associer correctement les bits aux bon sous blocs,
    # il faut modifier dans la classe et dans cette fonction


    # before_last_row = ([1] + [0] * (2 * t - 1)) * (m // 2) + [0] * t
    # pm[-2, :] = np.array(before_last_row)
    pm = [list(row) for row in pm]
    return ms(pm)


class ErrorLocatingCodeSp2:
    """Implements a special construction of EL codes over GF(2)
    m is the number of blocks, t is the number of bits in a single block"""
    def __init__(self, t, m, r1=1):
        self.m = m
        self.t = t
        self.q = 2
        self.r1 = r1
        self.k = m-r1
        self.length = self.m * self.t
        # nbrows x nbcols
        self.sub_paritymatrix_shape = self.k/self.t, t

        self.t0 = sqrt_int(self.t)
        self.m0 = sqrt_int(self.m)

        self.H = el_parity_matrix(self.t, self.m, r1=self.r1)
        self.sub_r = (self.H.nrows() - self.r1)/self.m
        self.sub_n = self.H.ncols()/self.m
        self.sub_k = self.sub_n - self.sub_r

        self.G = Matrix(kernel(self.H.transpose()).basis())
        self.msgspace = VectorSpace(GF(self.q), self.k)
        self.codewordspace = VectorSpace(GF(self.q), self.length)

    def random_msg(self):
        return self.msgspace.random_element()

    def random_codeword(self, return_block=False):
        cw = self.random_msg() * self.G
        if return_block:
            block = self.to_block(cw)
            return block
        else:
            return cw

    def syndrome(self, y):
        return self.H * y
    syn = syndrome

    def is_zerosyn(self, y):
        s = tuple(map(int, self.syn(y)))
        return np.allclose(s, 0)

    def sub_syndromes(self, y):
        """returns a tuple of syndrome localization status
        SB1 status is 1 if it is non zero,
        SB2 status is 0 if it is zero"""
        sub_r = self.sub_r
        l = self.m + self.r1 # sub syndromes + parity check bits
        s = tuple(map(int, self.syn(y)))
        sub_syns = np.zeros(l).astype(int)
        for i in range(l):
            sub_syn = s[i*sub_r: (i+1)*sub_r]
            # print(sub_syn)
            if np.allclose(sub_syn, 0):
                sub_syns[i] = 0
            else:
                sub_syns[i] = 1
        return sub_syns

    def print_allsyndromes(self):
        allerrors = self.codewordspace.list()
        for error in allerrors:
            s = self.syndrome(error)
            print_vec("Error : ", error, step=self.t, nocolor=True)
            print_vec("  Syn : ", s, step=self.sub_r, nocolor=True)
            print()

    def to_block(self, c):
        """method that transforms a codeword c into a 2D matrix block
        of size t0*m0 x t0*m0
        t0 = sqrt(t), m0 = sqrt(m)
        """

        block_sb = np.zeros((self.t0*self.m0, self.t0*self.m0)).astype(int)
        i = 1
        gen = ap.sqblock_gen_enum(block_sb, self.t0*self.m0, self.t0)
        for w0, w1, h0, h1 in gen:
            cw_sb = np.array(list(c[(i-1)*self.t: i*self.t]))
            block_sb[w0: w1, h0: h1] = cw_sb.reshape(self.t0, self.t0)
            i += 1
        return block_sb


    def to_vec(self, block):
        """transforms a block into its corresponding vector representation
        it is the inverse operation of to_block method"""
        c = np.zeros(self.t*self.m).astype(int)
        i = 0
        gen = ap.sqblock_gen_enum(block, self.t0*self.m0, self.t0)
        for w0, w1, h0, h1 in gen:
            block_sb = block[w0: w1, h0: h1]
            # print(block_sb)
            # print(c[i*self.t: (i+1)*self.t], i)
            c[i*self.t: (i+1)*self.t] = block_sb.flatten()

            i += 1
        c = self.codewordspace(list(c))
        return c


def test_ELC_class():
    code = ErrorLocatingCodeSp2(4, 4)
    code.print_allsyndromes()
    # print(code.H)
    #
    #
    # for i in range(10):
    #     cw = code.random_codeword()
    #
    #     block = code.to_block(cw)
    #     print_vec("", cw, step=4, endline=True)
    #     cw2 = code.to_vec(block)
    #
    #
    #     print_vec("", cw2, step=4, endline=True)
    #     print()
    #     assert cw == cw2


def main():
    test_ELC_class()


if __name__ == "__main__":
    sys.exit(main())
