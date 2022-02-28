from __future__ import print_function
import os
HOME = os.environ["HOME"]

import sys
import numpy as np
from sage.all import *
from myfinitefield import MyFiniteField


def sqrt_int(n):
    return int(np.sqrt(n))

def cw2bin(cw, mff):
    n, m = cw.length(), mff.m
    l = m * n
    binary = np.zeros(l)
    index = range(0, l, m)

    for i, ffelt in zip(index, cw):
        vec = mff.tovec(ffelt)
        # use lift to move from vector space to integers (as int64)
        binary[i:i+m] = np.array(lift(vec))

    return binary.astype(int)

def bin2cw(binary, mff):
    binary = binary.astype(int)
    elts = np.split(binary, len(binary)/mff.m)
    n = len(elts)
    vs = VectorSpace(mff.field, n)
    cw = map(mff.toff, elts)

    return vs(cw)

def cw2binblock(cw, mff):
    n, s = cw.length(), mff.m
    n0, s0 = sqrt_int(n), sqrt_int(s)
    l = n0*s0
    # print(n0, s0)
    block = np.zeros((l, l))
    # print(l)
    # print(block)

    for ind, symbol in enumerate(cw):
        indw, indh = ind/n0, ind%n0
        w0, h0 = indw*s0, indh*s0
        w1, h1 = w0 + s0, h0 + s0
        # print("n s ind = ", n, s, ind)
        # print(w0, w1, h0, h1)
        # print(w1, h1)

        vec = mff.tovec(symbol)

        # print("vec =")
        # print(vec)
        block_symbol = np.array(vec).reshape(s0, s0)
        # print("block symbol")
        # print(block_symbol)
        # print("room for symbol")
        # print(block[w0: w1, h0: h1])
        block[w0: w1, h0: h1] = block_symbol
        # print()
    # print(block)
    return block.astype(int)


def binblock2cw(block, mff):
    """square block codeword represented in binary mode
    to be converted into a codeword"""
    s0 = sqrt_int(mff.m)
    h = block.shape[1]
    n0 = h/s0
    # print("params = ", s0, n0)
    symbols = []
    indexes = range(0, n0*s0, s0)
    for indw in indexes:
        for indh in indexes:
            w0, h0 = indw, indh
            w1, h1 = w0 + s0, h0 + s0
            binblock_symbol = block[w0: w1, h0: h1]
            binsymbol_flat = binblock_symbol.flatten().astype(int)
            symbol = mff.toff(binsymbol_flat)
            # print(indw, indh)
            # print(binsymbol_flat)
            # print(symbol)
            symbols.append(symbol)

    vs = VectorSpace(mff.field, len(symbols))
    return vs(symbols)

def test_binblockconversion():
    n = 16
    s = 4
    print("n = ", n)
    print("s = ", s)
    mff = MyFiniteField(2, s)
    vs = VectorSpace(mff.field, n)

    # vec = vs.random_element(repr="ff")
    # print(vec)
    # block_cw = cw2binblock(vec, mff)
    # print(block_cw)
    # print("="*20)
    # cw = binblock2cw(block_cw, mff)
    # print(cw)
    # print(vec == cw)

    try:
        equal = True
        while equal:
            # vec is the initial vector to compare with
            vec = vs.random_element(repr="ff")
            binary = cw2bin(vec, mff)

            print(vec)
            print(binary)

            block_cw = cw2binblock(vec, mff)
            cw = binblock2cw(block_cw, mff)

            # vec2 = bin2cw(binary, mff)
            equal = (vec == cw)
    except KeyboardInterrupt:
        print("Interrupted by user")
        return


def test_bincwconversion():
    mff = MyFiniteField(2, 8)
    vs = VectorSpace(mff.field, 10)

    try:
        equal = True
        while equal:
            vec = vs.random_element(repr="ff")
            binary = cw2bin(vec, mff)

            print(vec)
            print(binary)

            vec2 = bin2cw(binary, mff)
            equal = (vec2 == vec)
    except KeyboardInterrupt:
        print("Interrupted by user")
        return
    # print(vec2==)

    # print(vec2)



def main():
    # test_random_error()
    # test_bincwconversion()
    test_binblockconversion()

if __name__ == "__main__":
    sys.exit(main())