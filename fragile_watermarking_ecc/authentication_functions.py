from __future__ import print_function
import os


import sys
import numpy as np
import arrayprocessing as ap
from bchcodes import bch_params


def el_loc(block_z, block_y, code):
    tamper_block = np.zeros_like(block_z)
    cw_z = code.to_vec(block_z)
    s = code.syn(cw_z)
    sub_r = code.sub_r
    m = code.m

    sub_syns = code.sub_syndromes(cw_z)
    iszero_parity_syn = sub_syns[-1] == 0

    gen = ap.sqblock_gen_enum(block_z, code.m0, code.t0)
    for i, coords in enumerate(gen):
        ww0, ww1, hh0, hh1 = coords
        if code.is_zerosyn(cw_z):
            pass
        else:
            sub_syn = sub_syns[i]
            if sub_syn != 0:
                tamper_block[ww0: ww1, hh0: hh1] = 255.

            if sub_syn == 0 and not iszero_parity_syn:
                tamper_block[ww0: ww1, hh0: hh1] = 255.

    return tamper_block


def el_loc_nosyn(block_z, block_y, code):
    tamper_block = np.zeros_like(block_z)
    cw_z = code.to_vec(block_z)
    s = code.syn(cw_z)
    sub_r = code.sub_r
    m = code.m

    sub_syns = code.sub_syndromes(cw_z)
    iszero_parity_syn = sub_syns[-1] == 0

    gen = ap.sqblock_gen_enum(block_z, code.m0, code.t0)
    for i, coords in enumerate(gen):
        ww0, ww1, hh0, hh1 = coords
        if code.is_zerosyn(cw_z):
            pass
        else:
            sub_syn = sub_syns[i]
            if sub_syn != 0:
                tamper_block[ww0: ww1, hh0: hh1] = 255.

            # if sub_syn == 0 and not iszero_parity_syn:
            #     tamper_block[ww0: ww1, hh0: hh1] = 255.

    return tamper_block


def bch_dec(block_z, block_y, code):
    tamper_block = np.zeros_like(block_z)
    s = code.t
    n = code.m
    s0 = code.t0
    n0 = code.m0

    n_bch, k_bch, t_bch = bch_params[(s, n)]
    diff_pos = np.where(block_z != block_y)
    t_bch2 = len(diff_pos[0])
    if t_bch2 <= t_bch:
        tamper_block[block_z != block_y] = 255.
        return tamper_block
    else:
        return np.zeros_like(block_z) + 255.


def rs_syn(block_z, block_y, code):
    tamper_block = np.zeros_like(block_z)
    s0 = code.t0

    gen = ap.sqblock_gen_enum2(block_z, s0)
    for i, coords in enumerate(gen):
        ww0, ww1, hh0, hh1 = coords
        sub_block_z = block_z[ww0: ww1, hh0: hh1]
        sub_block_y = block_y[ww0: ww1, hh0: hh1]
        if not np.allclose(sub_block_z, sub_block_y):
            return tamper_block + 255.

    return tamper_block


def rs_loc(block_z, block_y, code):
    tamper_block = np.zeros_like(block_z)
    s = code.t
    n = code.m
    s0 = code.t0
    n0 = code.m0

    max_nb_of_errors = int(np.floor((n - 1) / 2.))
    nb_of_errors = 0  # number of erroneous symbols

    gen = ap.sqblock_gen_enum(block_z, n0, s0)
    for i, coords in enumerate(gen):
        ww0, ww1, hh0, hh1 = coords
        sub_block_z = block_z[ww0: ww1, hh0: hh1]
        sub_block_y = block_y[ww0: ww1, hh0: hh1]
        if not np.allclose(sub_block_z, sub_block_y):
            nb_of_errors += 1
            tamper_block[ww0: ww1, hh0: hh1] = 255.

    if nb_of_errors <= max_nb_of_errors:
        return tamper_block
    else:
        return np.zeros_like(block_z) + 255.


def rs_dec(block_z, block_y, code):
    tamper_block = np.zeros_like(block_z)
    s = code.t
    n = code.m
    s0 = code.t0
    n0 = code.m0

    max_nb_of_errors = int(np.floor((n - 1) / 2.))
    nb_of_errors = 0  # number of erroneous symbols

    gen = ap.sqblock_gen_enum(block_z, n0, s0)
    for i, coords in enumerate(gen):
        ww0, ww1, hh0, hh1 = coords
        sub_block_z = block_z[ww0: ww1, hh0: hh1]
        sub_block_y = block_y[ww0: ww1, hh0: hh1]
        if not np.allclose(sub_block_z, sub_block_y):
            nb_of_errors += 1
            tamper_block[block_z != block_y] = 255.

    if nb_of_errors <= max_nb_of_errors:
        return tamper_block
    else:
        return np.zeros_like(block_z) + 255.


auth_modes = {
    "RSSYN"     : rs_syn,
    "RSLOC"     : rs_loc,
    "RSDEC"     : rs_dec,

    "ELLOC"     : el_loc,
    "ELLOCNOSYN": el_loc_nosyn,

    "BCHDEC"    : bch_dec,

    "QISVD"     : None,
    "PFLOC"     : None,  # perfect LOC
}


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
