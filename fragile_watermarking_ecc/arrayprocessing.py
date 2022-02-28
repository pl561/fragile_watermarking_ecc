from __future__ import print_function
import os
HOME = os.environ["HOME"]

import sys
import numpy as np



def block_mswaps2d(array, pos_dict, size):
    for pos1, pos2 in pos_dict.items():
        block_swap2d(array, pos1, pos2, size)


def block_swap2d(array, pos1, pos2, size):
    sw, sh = size
    w0, h0 = pos1
    w1, h1 = pos2

    b0 = np.copy(array[w0:w0+sw,h0:h0+sh])
    b1 = np.copy(array[w1:w1+sw,h1:h1+sh])

    # print("=====")
    # print(array)
    # print(size)
    # print(w0, h0)
    # print(w0+sw, h0+sh)
    # print(array[4:5,0:4])
    # print(array[w0:w0+sw,h0:h0+sh])
    # print(b1)

    array[w0:w0+sw,h0:h0+sh] = b1
    array[w1:w1+sw,h1:h1+sh] = b0


def block_swap2d_lsb(array, pos1, pos2, size):
    arr_lsb = array%2
    array -= arr_lsb
    block_swap2d(arr_lsb, pos1, pos2, size)
    array += arr_lsb
    return array


def test_swap_lsb():
    arr = np.arange(81).reshape(9, 9)
    print(arr)
    arr = block_swap2d_lsb(arr, (0, 4), (1, 0), (1, 4))
    print(arr)
    arr = block_swap2d_lsb(arr, (0, 4), (1, 0), (1, 4))
    print(arr)

def test_mswaps2d():
    pos_dict = {
        (0, 4) : (1, 0),
        (2, 4) : (3, 0),
        (4, 4) : (5, 0),
        (6, 4) : (7, 0),
    }
    size = (1, 4)
    block_config = pos_dict, size
    array = np.arange(64).reshape(8, 8)
    print(array)
    block_mswaps2d(array, pos_dict, size)
    print(array)

def block_permutations(block, n):
    """symbols in a cw are embedded in line in a block of n x s pixels
       we have s = s1^2, n = s, n code length, s the extension degree
       a line is transformed into a square sub-block of s1 x s1 pixels"""
    s1 = int(np.sqrt(n))
    size = (1, s1)

    for k in range(0, n, s1):
        for i in range(s1):
            for j in range(i+1, s1):
                pos1 = (i+k, j*s1)
                pos2 = (j+k, i*s1)
                # print("{} --> {}".format(str(pos1), str(pos2)))
                block_swap2d(block, pos1, pos2, size)
                # cpt += 1
            # if cpt >= 3:
            #     return

# https://anandology.com/python-practice-book/iterators.html
# iterator version with class attributes such as fixed params s0, n0
# def sqblock_gen(block, n0, s0, strict_size=False):
#     """generator that iterates on square blocks
#        inside a square block
#        block is composed of n0 x n0 sub-blocks of size s0 x s0
#        """
#     gen = sqblock_gen_enum(block, n0, s0)
#     for w0, w1, h0, h1 in gen:
#         b = block[w0: w1, h0: h1]
#         if b.shape == (s0, s0):
#             yield b

def sqblock_gen_enum(block, n0, s0):
    """generator that iterates on non-overlaping square blocks
       inside a square block
       block is composed of n0 x n0 sub-blocks of size s0 x s0
       if strict_size is True, the generator will only yield blocks with
       exact size s0 x s0
       """
    indexes = range(0, n0*s0, s0)
    for indw in indexes:
        for indh in indexes:
            w0, h0 = indw, indh
            w1, h1 = w0 + s0, h0 + s0
            b = block[w0: w1, h0: h1]
            if b.shape == (s0, s0):
                yield w0, w1, h0, h1


def sqblock_gen_enum2(block, s0):
    """
    generator that iterates over a 2D array and yield non-overlaping
    blocks of size s0 x s0
    if the image is not a multiple of s0 x s0, the image borders are ignored
    """
    w, h = block.shape
    indexes_h = range(0, h, s0)
    indexes_w = range(0, w, s0)
    for indw in indexes_w:
        for indh in indexes_h:
            w0, h0 = indw, indh
            w1, h1 = w0 + s0, h0 + s0
            b = block[w0: w1, h0: h1]
            if b.shape == (s0, s0):
                yield w0, w1, h0, h1


def window_sliding(block, ws=3):
    """
    implements a sliding square window iterator
    :param block: a 2D array to iterate through
    :param ws: window size, odd integer
    :yields: window coordinates
    """
    assert ws%2 == 1, "ws must be an odd number"
    w, h = block.shape
    d = (ws-1)/2
    for indw in range(w):
        for indh in range(h):
            w0, h0 = indw - d, indh - d
            if w0 < 0:
                w0 = 0
            if h0 < 0:
                h0 = 0
            w1, h1 = indw + d + 1, indh + d + 1
            yield w0, w1, h0, h1




def process_byblock(array, func, blocksize=8):
    """Apply a function func to every sub block (of size blocksize) of array"""
    processed_array = np.zeros_like(array, dtype=float)
    gen = sqblock_gen_enum2(array, blocksize)
    for w0, w1, h0, h1 in gen:
        processed_array[w0: w1, h0: h1] = func(array[w0: w1, h0: h1])
    return processed_array


def test_sqblock_iter():
    n0, s0 = 6, 5
    side_size = n0 * s0
    l = side_size**2
    big_block = np.arange(l).reshape(side_size, side_size)
    print(big_block)
    # for w0, w1, h0, h1 in sqblock_gen_enum(big_block, n0, s0):
    #     print(w0, w1, h0, h1, "\n")
    #     print(big_block[w0: w1, h0: h1])
    # print(big_block)

    for w0, w1, h0, h1 in sqblock_gen_enum(big_block, n0, 4):
        print(w0, w1, h0, h1, "\n")
        print(big_block[w0: w1, h0: h1])
        # print(big_block)

def test_block_permutations():
    n = 9
    s1 = int(np.sqrt(n))
    block = np.arange(n*n).reshape(n, n)
    print(block)
    block_permutations(block, n)
    print(block)
    block_permutations(block, n)
    print(block)

def test_sliding_window():
    error_map = np.arange(1, 10).reshape(3, 3)
    gen = window_sliding(error_map, ws=3)
    for w0, w1, h0, h1 in gen:
        b = error_map[w0: w1, h0: h1]
        print(b)
        print("----")

def main():
    # test_block_permutations()
    # test_sqblock_iter()
    test_sliding_window()

if __name__ == "__main__":
    sys.exit(main())