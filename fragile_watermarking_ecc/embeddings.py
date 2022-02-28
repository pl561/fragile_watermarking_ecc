from __future__ import print_function
import os

HOME = os.environ["HOME"]
import sys
import numpy as np
import pandas as pd
import cv2

import pysnooper


from ecc import sqrt_int
import arrayprocessing as ap
import metrics as mtc
from authentication_functions import auth_modes
from qimdwt import QIMDWT

from pathtools import EPath

# @pysnooper.snoop()
def embedding_lsbsp(img_x, code):
    """LSB spatial embedding"""
    img_y = np.copy(img_x).astype(int)
    h, w = img_y.shape  # height (as nb of line), width (as nb of cols)

    k, n = code.k, code.length
    s = code.t
    n = code.m
    q = code.q

    n0, s0 = sqrt_int(n), sqrt_int(s)
    min_wh = min(w, h)
    # faster operation
    img_y[: min_wh, : min_wh] -= img_y[: min_wh, : min_wh] % 2

    gen = ap.sqblock_gen_enum(img_x, min_wh, n0 * s0)
    for w0, w1, h0, h1 in gen:
        cw_x = code.random_codeword()
        block_cw = code.to_block(cw_x)
        # img_y[w0: w1, h0: h1] -= img_y[w0: w1, h0: h1] % 2 # not fast
        img_y[w0: w1, h0: h1] += block_cw

    return img_y


def detection_lsbsp(img_z, img_y, code, auth_mode):
    """LSB spatial detection"""
    img_z = np.copy(img_z).astype(int)
    tamper_map = np.zeros(img_z.shape, dtype=np.float64)
    h, w = img_z.shape  # height (as nb of line), width (as nb of width)

    k, n, s = code.k, code.m, code.t

    q = code.q

    s0 = sqrt_int(s)
    n0 = sqrt_int(n)

    auth_func = auth_modes[auth_mode]

    nb_cw = 0
    nb_erroneous_cw = 0
    # number of codewords that have been localized or corrected
    nb_controlled_cw = 0

    gen = ap.sqblock_gen_enum(img_z, min(h, w), n0 * s0)
    for w0, w1, h0, h1 in gen:
        block_y = img_y[w0: w1, h0: h1] % 2
        block_z = img_z[w0: w1, h0: h1] % 2

        tamper_block = auth_func(block_z, block_y, code)
        tamper_map[w0: w1, h0: h1] = tamper_block

    cw_stats = nb_cw, nb_erroneous_cw, nb_controlled_cw
    return tamper_map, cw_stats


class TamperingLocalizationLSBSP:
    def __init__(self, code, auth_mode):
        self.code = code
        self.auth_mode = auth_mode

    def embedding(self, img_x):
        embedding_lsbsp(img_x, self.code)

    def detection(self, img_z, img_y):
        return detection_lsbsp(img_z, img_y, self.code, self.auth_mode)



def embedding_qimdwt(img_x, code,
                     delta, level, detail_level, band_name,
                     wt_type="db4", wt_mode='periodization'):
    """SWT QIM embedding"""
    img_y = np.copy(img_x).astype(int)
    h, w = img_y.shape  # height (as nb of line), width (as nb of cols)

    k, n = code.k, code.length
    s = code.t
    n = code.m
    q = code.q

    n0, s0 = sqrt_int(n), sqrt_int(s)

    # QIM DWT band extraction
    qimdwt_inst = QIMDWT(img_x, delta,
                         level, detail_level, band_name,
                         wt_type=wt_type,
                         mode=wt_mode)
    qimdwt_inst.decompose()
    band_x = qimdwt_inst.get_band()
    w, h = qimdwt_inst.get_band_shape()

    # create a binary plane to embed in band_x
    binary_plane = np.zeros_like(band_x)
    min_wh = min(w, h)
    gen = ap.sqblock_gen_enum(band_x, min_wh, n0 * s0)



    for w0, w1, h0, h1 in gen:
        block_cw = code.random_codeword(return_block=True)
        bp_tmp = binary_plane[w0: w1, h0: h1].astype(int)
        # print(block_cw)
        # print(bp_tmp)
        # print(block_cw.shape, bp_tmp.shape)
        # print(binary_plane.shape, min_wh)
        binary_plane[w0: w1, h0: h1] = block_cw

    qimdwt_inst.embed(binary_plane)
    img_y = qimdwt_inst.recompose().astype(int)

    return img_y


def detection_qimdwt(img_z, img_y, code, auth_mode,
                     delta, level, detail_level, band_name,
                     wt_type="db4", wt_mode='periodization'):
    """DWT QIM detection"""
    img_z = np.copy(img_z).astype(int)
    # tamper_map = np.zeros(img_z.shape, dtype=np.float64)
    # h, w = img_z.shape  # height (as nb of line), width (as nb of width)

    k, n, s = code.k, code.m, code.t

    q = code.q

    s0 = sqrt_int(s)
    n0 = sqrt_int(n)

    auth_func = auth_modes[auth_mode]

    nb_cw = 0
    nb_erroneous_cw = 0
    # number of codewords that have been localized or corrected
    nb_controlled_cw = 0


    # QIM DWT band extraction
    qimdwt_inst_y = QIMDWT(img_y, delta,
                           level, detail_level, band_name,
                           wt_type=wt_type,
                           mode=wt_mode)
    qimdwt_inst_y.decompose()
    band_y = qimdwt_inst_y.get_band()
    band_detected_y, bp_y = qimdwt_inst_y.detect()

    qimdwt_inst_z = QIMDWT(img_z, delta,
                           level, detail_level, band_name,
                           wt_type=wt_type,
                           mode=wt_mode)
    qimdwt_inst_z.decompose()
    band_z = qimdwt_inst_z.get_band()
    band_detected_z, bp_z = qimdwt_inst_z.detect()
    w, h = qimdwt_inst_z.get_band_shape()

    tm_band_z = np.zeros(band_z.shape, dtype=np.float64)

    min_wh = min(w, h)
    gen = ap.sqblock_gen_enum(band_z, min_wh, n0 * s0)

    # print(bp_y)
    # print(bp_z)

    for w0, w1, h0, h1 in gen:
        # print(w0, w1, h0, h1)
        block_y = bp_y[w0: w1, h0: h1]
        block_z = bp_z[w0: w1, h0: h1]

        tamper_block = auth_func(block_z, block_y, code)
        tm_band_z[w0: w1, h0: h1] = tamper_block

    # =====================================================

    tmb_z = tm_band_z
    tmb_z = apply_threshold(tmb_z)

    ##########################
    # tamper band processing #
    ##########################


    filtering_mode = "WS"

    if filtering_mode == "WS":
        tamper_map2 = ws_filter_mode(tmb_z, qimdwt_inst_z)
    elif filtering_mode == "classicfiltering":
        tamper_map2 = classic_filter_mode(tmb_z, qimdwt_inst_z)
    else:
        tamper_map2 = np.zeros_like(img_z)
        raise ValueError("unknown filtering mode")



    sample = False
    if sample:
        print(img_z.shape)
        print(tamper_map2.shape)

        # fname_out = "/tmp/recomp_fromzero_{}_{}.png"
        # fname_out = fname_out.format(qimdwt_inst_z.wt_type,
        #                              qimdwt_inst_z.mode)



        # fname_out0 = "/tmp/sample/tm_dwt_z.png"
        # fname_out1 = "/tmp/tm_sp_z.png"

        # band_y_rescaled = rescale(band_y)
        # band_z_rescaled = 255-rescale(band_z*1.0)
        dwt_diff = np.abs(band_y-band_z)
        sp_diff = np.abs(img_y-img_z)


        # top_out = np.hstack((tmb_z, tm_band_z))
        # bot_out = np.hstack((band_z, dwt_diff))
        # bands = np.vstack((top_out, bot_out))
        #
        # top_out = np.hstack((img_y, img_z))
        # bot_out = np.hstack((tamper_map0, tamper_map2))
        # img_out = np.vstack((top_out, bot_out))

        # cv2.imwrite(fname_out0, bands)
        # cv2.imwrite(fname_out1, img_out)

        sample_dir = EPath("/tmp/sample2")
        sample_dir.mkdir()
        sample_dir.join("img_y.png").imwrite(img_y)
        sample_dir.join("img_z.png").imwrite(img_z)
        sample_dir.join("band_y.png").imwrite(band_y)
        sample_dir.join("band_z.png").imwrite(band_z)
        # sample_dir.join("tm_sp_y.png").imwrite(_y)
        sample_dir.join("dwt_diff.png").imwrite(dwt_diff)
        sample_dir.join("tm_dwt_z.png").imwrite(tmb_z)
        sample_dir.join("tm_sp_z.png").imwrite(tamper_map2)


    cw_stats = nb_cw, nb_erroneous_cw, nb_controlled_cw
    return tamper_map2, cw_stats

def rescale(array):
    array = array.astype(float)
    mini, maxi = array.min(), array.max()
    rescaled = (array-mini)/maxi*255
    return rescaled.astype(int)

def apply_threshold(arr, t=1e-1):
    arr[arr > t] = 255.
    arr[arr < t] = 0.
    return arr

def apply_threshold_frommax(arr, t=0.1):
    """search for the max gs value and apply a threshold from t=0.9*max"""
    maxi = arr.max()
    arr = apply_threshold(arr, t=maxi*(1-t))
    return arr

def ws_filter_mode(tmb, qimdwt_inst):
    tmb = mtc.window_sliding_binarymap(tmb, nb_pixel=3, ws=3)


    tamper_map = qimdwt_inst.recompose_withnone(tmb)
    tamper_map = apply_threshold(tamper_map)
    tamper_map= mtc.window_sliding_binarymap(tamper_map, nb_pixel=3, ws=3)
    return tamper_map


def classic_filter_mode(tmb, qimdwt_inst):
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    # tmb = cv2.blur(tmb, (5, 5))
    # tm_band_z = cv2.blur(tm_band_z, (5, 5))


    tmb = cv2.dilate(tmb, kernel=kernel3, iterations=2)
    # tmb = cv2.blur(tmb, (5, 5))
    # tmb = apply_threshold_frommax(tmb, t=0.5)
    # tmb = apply_threshold(tmb)
    tmb = cv2.erode(tmb, kernel=kernel3, iterations=3)
    tmb = cv2.dilate(tmb, kernel=kernel3, iterations=2)
    # tmb = cv2.blur(tmb, (3, 3))
    # tmb = apply_threshold(tmb)


    # tmb = cv2.dilate(tmb, kernel=kernel3, iterations=1)

    tamper_map0 = qimdwt_inst.recompose_withnone(tmb)

    # tamper map processing, dtype : float
    threshold = 1e-1
    tamper_map = apply_threshold(np.copy(tamper_map0))
    tamper_map2 = tamper_map
    tamper_map2 = cv2.blur(tamper_map2, (5, 5))
    # tamper_map2 = cv2.blur(tamper_map2, (5, 5))
    # tamper_map2 = cv2.dilate(tamper_map2, kernel3)
    tamper_map2 = apply_threshold(tamper_map2)
    # tamper_map2 = cv2.erode(tamper_map2, kernel5)
    # threshold = 1e-2
    # tamper_map2 = apply_threshold(tamper_map2)
    return tamper_map2



class TamperingLocalizationQIMDWT:
    def __init__(self, code, auth_mode,
                 delta, level,
                 detail_level, band_name,
                 wt_type="db4", wt_mode='periodization'):
        self.code = code
        self.auth_mode = auth_mode
        self.delta = delta
        self.level = level
        self.wt_type = wt_type
        self.wt_mode = wt_mode
        self.detail_level = detail_level
        self.band_name = band_name

    def embedding(self, img_x):
        return embedding_qimdwt(img_x, self.code,
                                self.delta, self.level,
                                self.detail_level, self.band_name,
                                wt_type=self.wt_type,
                                wt_mode=self.wt_mode)

    def detection(self, img_z, img_y):
        return detection_qimdwt(img_z, img_y, self.code, self.auth_mode,
                                self.delta, self.level,
                                self.detail_level, self.band_name,
                                wt_type=self.wt_type,
                                wt_mode=self.wt_mode)

def main():
    pass

if __name__ == "__main__":
    sys.exit(main())