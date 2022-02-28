"""Implementation from Qi et al.'s article, 2015"""


from __future__ import print_function
import os

HOME = os.environ["HOME"]
import sys
import numpy as np
import pandas as pd
import cv2
import pywt

from ecc import sqrt_int
import arrayprocessing as ap
import attacks as atk
from metrics import compute_psnr, \
    binarymap, metric_m1, window_sliding_binarymap, confusion_map
import metrics as mtc
from qimdwt import QIMDWT

from pathtools import EPath


# JPEG_QUANTIZATION_MATRIX
JQM = np.array([
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101, 
    72, 92, 95, 98, 112, 100, 103,  99,
]).reshape(8, 8).astype(float)



def generate_quantized_image(img):
    """
    Quantize the image with jpeg quantization matrix JQM
    :param img: an image
    :return: a quantized image q_img
    """
    img = img.astype(float)
    q_img = np.zeros_like(img)
    gen8 = ap.sqblock_gen_enum2(img, 8)
    for w0, w1, h0, h1 in gen8:
        block = img[w0: w1, h0: h1]
        q_img[w0: w1, h0: h1] = np.round(block/JQM) * JQM
    return q_img


def extract_cw_pblock(block_y4):
    gen2 = ap.sqblock_gen_enum2(block_y4, 2)
    s_values = np.zeros(4)
    i = 0
    for www0, www1, hhh0, hhh1 in gen2:
        block_y2 = block_y4[www0: www1, hhh0: hhh1]
        u, s, vh = np.linalg.svd(block_y2, full_matrices=False)
        s_values[i] = s[0]
        i += 1

    B1, B2, B3 = 0, 0, 0
    if s_values[1] >= s_values[2]:
        B1 = 1
    if s_values[2] >= s_values[3]:
        B2 = 1
    if s_values[3] >= s_values[1]:
        B3 = 1

    PBlock_i = B1 + B2 + B3
    CW_i = PBlock_i % 2
    return CW_i, PBlock_i


def generate_cw_pblock_maps(img_x):
    """
    generates a CW map and a PBlock map from the quantized image q_img
    :param img_x: host image
    :param q_img: quantized image
    :return: CW_map, Pblock_map
    """
    q_img = generate_quantized_image(img_x)
    CW_map = np.zeros_like(q_img)
    PBlock_map = np.zeros_like(q_img)

    gen4 = ap.sqblock_gen_enum2(img_x, 4)
    one4 = np.ones((4, 4))

    for w0, w1, h0, h1 in gen4:
        block4 = img_x[w0: w1, h0: h1]
        CW, PBlock = extract_cw_pblock(block4)
        CW_map[w0: w1, h0: h1] = one4 * CW
        PBlock_map[w0: w1, h0: h1] = one4 * PBlock

    return CW_map, PBlock_map

def generate_iw_map(CW_map_shape):
    """
    generates a random IW map
    :param IW_map_shape:
    :return:
    """
    IW_map = np.zeros(CW_map_shape)
    gen4 = ap.sqblock_gen_enum2(IW_map, 4)
    one4 = np.ones((4, 4))
    for w0, w1, h0, h1 in gen4:
        rd_bit = np.random.choice([0, 1], 1)
        if rd_bit == 1:
            IW_map[w0: w1, h0: h1] = one4
    return IW_map


# def generate_cw_watermark(CW_map):
#     """
#     generates the content dependent watermark CW_seq sequence
#     :param CW_map:
#     :return: CW_seq
#     """
#     CW_seq = []
#     gen4 = ap.sqblock_gen_enum2(CW_map, 4)
#     for w0, w1, h0, h1 in gen4:
#         CW_seq.append(CW_map[w0, h0])
#
#     # def func(block):
#     #     return block[0, 0]
#     # CW_seq2 = ap.process_byblock(CW_map, func, 4)
#     #
#     # print(CW_seq2)
#     # assert np.allclose(CW_seq, CW_seq2)
#
#     return np.array(CW_seq)


# def seq2map4(map_shape, seq):
#     arr_map = np.zeros(map_shape)
#     one4 = np.ones((4, 4))
#     gen4 = ap.sqblock_gen_enum2(arr_map, 4)
#     i = 0
#     for w0, w1, h0, h1 in gen4:
#         arr_map[w0: w1, h0: h1] = one4 * seq[i]
#         i += 1
#     return arr_map


# def generate_pblock(PBlock_map):
#     """
#     generates the content dependent watermark CW_seq sequence
#     :param CW_map:
#     :return: CW_seq
#     """
#     PBlock_seq = []
#     gen4 = ap.sqblock_gen_enum2(PBlock_map, 4)
#     for w0, w1, h0, h1 in gen4:
#         PBlock_seq.append(PBlock_map[w0, h0])
#     return np.array(PBlock_seq)
#
#
# def generate_iw_watermark(seq_len):
#     """
#     generates the content independent watermark IW_seq
#     :param seq_len: binary length of IW_seq
#     :return: IW_seq
#     """
#     IW_seq = np.random.choice((0, 1), seq_len)
#     return IW_seq
#
#
# def compute_secure_watermark(CW_map):
#     """
#     computes the secure watermark SW_seq
#     :param CW_map: content dependent watermark as a 2D array
#     :return: SW_seq
#     """
#     CW_seq = generate_cw_watermark(CW_map)
#     IW_seq = generate_iw_watermark(len(CW_seq))
#     SW_seq = (CW_seq + IW_seq) % 2
#     return SW_seq, IW_seq

def embedding_qi2015singular(img_x, SW_map):
    """
    embeds a watermark inside the host image
    img_y is the watermarked image
    :param img_x: host image
    :return: img_y
    """
    img_x = img_x.astype(float)
    img_y = np.copy(img_x)
    coord = 0, 0
    CW_map, PBlock_map = generate_cw_pblock_maps(img_x)

    gen4 = ap.sqblock_gen_enum2(img_x, 4)
    for w0, w1, h0, h1 in gen4:
        block4 = img_x[w0: w1, h0: h1]
        # CW = CW_map[w0, h0]
        PBlock = PBlock_map[w0, h0]
        q = 11 + 2 * PBlock

        # DWT on block4
        dec_obj = list(pywt.wavedec2(block4, "haar", level=1))
        LL = dec_obj[0][coord]
        LLq = np.round(LL/q)
        SW_coeff = int(SW_map[w0: w1, h0: h1][0, 0])
        assert SW_coeff in [0, 1]
        if LLq % 2 == SW_coeff:
            dec_obj[0][coord] = LLq * q
        else:
            dec_obj[0][coord] = LLq * q + q
        modified_block4 = pywt.waverec2(dec_obj, "haar")

        img_y[w0: w1, h0: h1] = modified_block4

    return img_y.astype(int)



def authentication_qi2015singular(img_z, IW_map):
    """
    extracts the error watermark EW_map and builds the error map IW_map
    :param img_z: modified image
    :param IW_map: computed from the secret key, it is transmitted to the
    receiver
    :return:
    """
    coord = 0, 0
    ones4 = np.ones((4, 4))
    EW_map = np.zeros_like(img_z)
    CW_map_z, PBlock_map_z = generate_cw_pblock_maps(img_z)


    # SW_map_z = compute_secure_watermark(CW_map_z)
    SW_map_z = (CW_map_z + IW_map) % 2
    gen4 = ap.sqblock_gen_enum2(img_z, 4)

    for w0, w1, h0, h1 in gen4:
        block4 = img_z[w0: w1, h0: h1]
        PBlock = PBlock_map_z[w0, h0]
        q = 11 + 2 * PBlock
        dec_obj = list(pywt.wavedec2(block4, "haar", level=1))
        LL = dec_obj[0][coord]
        LLq = np.round(LL/q)

        if LLq % 2 == 1:
            EW_map[w0: w1, h0: h1] = ones4

    # error map, or tamper map
    EM_map = np.abs(EW_map-SW_map_z)
    # STErrorMap
    ST_EM_map5 = window_sliding_binarymap(EM_map * 255., nb_pixel=5, ws=3)
    # return EM_map * 255., CW_map_z, PBlock_map_z

    return ST_EM_map5, CW_map_z, PBlock_map_z

class TamperingLocalization_qi2015singular:
    def __init__(self, img_x):
        """
        the use of img_x is redundant in this class
        img_x is also an argument of the embedding method
        :param img_x:
        """
        self.CW_map, self.PBlock_map = generate_cw_pblock_maps(img_x)
        self.IW_map = generate_iw_map(self.CW_map.shape)
        self.SW_map = (self.CW_map + self.IW_map) % 2

    def embedding(self, img_x):
        """returns the watermarked image img_y"""
        return embedding_qi2015singular(img_x, self.SW_map)

    def detection(self, img_z, img_y=None):
        # default output expected in benchmark function is the tamper map+tuple
        ST_EM5, CW_z, PBl_z = authentication_qi2015singular(img_z, self.IW_map)
        self.CW_z, self.PBl_z = CW_z, PBl_z
        return ST_EM5, (0, 0, 0)


def test_functions():
    from imagedatabase import get_tampered_realistic_png
    from imagedatabase import get_tampered_realistic_png_resized
    mydataset = get_tampered_realistic_png_resized()
    nb_image = 1 # [: nb_image]
    fname_pairs = list(zip(*mydataset))

    fname_x, fname_t = fname_pairs[2]

    ep_fname_x, ep_fname_t = EPath(fname_x), EPath(fname_t)
    # ep_fname_x = EPath(HOME).join("Images/lena.png")
    print(ep_fname_x.stem())
    img_x = ep_fname_x.imread_gs_int()


    print(img_x.shape)

    tl_qi = TamperingLocalization_qi2015singular(img_x)
    CW_map = tl_qi.CW_map
    PBlock_map = tl_qi.PBlock_map

    print("Embedding...")
    img_y = tl_qi.embedding(img_x)
    print("Done.")


    img_t = ep_fname_t.imread_gs_int()

    apply_tampering = True
    apply_attack = False

    print("Apply tampering : {}".format(apply_tampering))
    print("Apply attack    : {}".format(apply_attack))

    img_z = img_y

    if apply_tampering:
        img_z, tamper_map = atk.apply_tamper_pattern(img_z, img_x, img_t)
        print("Applied tampering")
    else:
        tamper_map = np.zeros_like(img_x)

    if apply_attack:
        av = 95
        img_z = atk.jpeg_compression(img_z, av)
        print("Attack value : {}".format(av))
        print("Applied attack")

    print("Authentication...")
    # EM_map, CW_map_z, PBlock_map_z = tl_qi.detection(img_z)
    EM_map, _ = tl_qi.detection(img_z)
    CW_map_z, PBlock_map_z = tl_qi.CW_map, tl_qi.PBlock_map
    print("Done.")

    ws = 3
    WS_EM_map3 = window_sliding_binarymap(EM_map, nb_pixel=3, ws=ws)
    # WS_EM_map4 = window_sliding_binarymap(EM_map, nb_pixel=4, ws=ws)
    WS_EM_map5 = window_sliding_binarymap(EM_map, nb_pixel=5, ws=ws)

    conf_map3 = confusion_map(tamper_map, WS_EM_map3)
    conf_map5 = confusion_map(tamper_map, WS_EM_map5)

    prec3 = mtc.compute_ppv(mtc.confusion_matrix(tamper_map, WS_EM_map3))
    prec5 = mtc.compute_ppv(mtc.confusion_matrix(tamper_map, WS_EM_map5))

    tamper_data3 = mtc.ConfusionData(tamper_map, WS_EM_map3)
    print(tamper_data3.compute_metrics())

    # ws = 5
    # WS_EM_map3 = window_sliding_binarymap(EM_map, nb_pixel=7, ws=ws)
    # WS_EM_map4 = window_sliding_binarymap(EM_map, nb_pixel=4, ws=ws)
    # WS_EM_map5 = window_sliding_binarymap(EM_map, nb_pixel=16, ws=ws)

    psnr_val = compute_psnr(img_x, img_y)
    print("PNSR = {} dB".format(psnr_val))
    print("M1   = {}".format(metric_m1(EM_map)))
    # print("Prec3   = {}".format(prec3))
    # print("Prec5   = {}".format(prec5))

    # cmp_pblock_map = np.allclose(PBlock_map, PBlock_map_z)
    # cmp_cw_map = np.allclose(CW_map, CW_map_z)

    q_img_x = generate_quantized_image(img_x)
    q_img_z = generate_quantized_image(img_z)

    # print("CW     eq :", cmp_cw_map)
    # print("PBlock eq :", cmp_pblock_map)


    # ep_fname_x.add_after_stem("tm").replace_parents("/tmp").imwrite(EM_map)
    # ep_fname_x.add_after_stem("w").replace_parents("/tmp").imwrite(img_y)

    parent = EPath("/tmp/test")
    parent.mkdir()
    p2 = parent.join("tm_ws")
    p2.mkdir()
    ep_test = EPath("imagetest.png").sc_rp(parent)



    ep_test.sc_aas("qimg").imwrite(binarymap(q_img_x, q_img_z))
    # ep_test.sc_aas("cw").imwrite(binarymap(CW_map, CW_map_z))
    # ep_test.sc_aas("pblock").imwrite(binarymap(PBlock_map, PBlock_map_z))

    ep_test.sc_rp(p2).sc_aas("tm").imwrite(EM_map)
    ep_test.sc_rp(p2).sc_aas("wstm3").imwrite(WS_EM_map3)
    # ep_test.sc_rp(p2).sc_aas("wstm4").imwrite(WS_EM_map4)
    ep_test.sc_rp(p2).sc_aas("wstm5").imwrite(WS_EM_map5)

    ep_test.sc_rp(p2).sc_aas("cm3").imwrite(conf_map3)
    ep_test.sc_rp(p2).sc_aas("cm5").imwrite(conf_map5)

    ep_test.sc_aas("w").imwrite(img_y)


def main():
    np.set_printoptions(precision=3, suppress=True)
    test_functions()


if __name__ == "__main__":
    sys.exit(main())