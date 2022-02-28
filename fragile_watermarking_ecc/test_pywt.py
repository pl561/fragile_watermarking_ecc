from __future__ import print_function
import sys

import numpy as np
import pywt
import cv2
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

from wavelets import Wavelets
from qim import QIM
from qimdwt import QIMDWT
import metrics as mtc
from attacks import image_processings, apply_tamper_pattern
from pathtools import EPath
from imagedatabase import get_tampered_realistic_png

# deltas = range(8, 30)
# nb_rep = 10
qimdwt_results = [
 0.04443359, 0.04321289, 0.04467773, 0.02319336, 0.,     0.02490234,
 0.04785156, 0.,     0.,     0.,     0.,     0.,
 0.,     0.01489258, 0.,     0.01245117, 0.01391602, 0.00439453,
 0.,     0.,     0.00341797, 0.00317383, 0.,     0.00439453,
 0.00268555, 0.,     0.0012207,  0.,     0.,     0.,
 0.00048828, 0.0012207, 0.,     0.00048828, 0.,     0.00048828,
 0.0012207, 0.00097656, 0.00024414, 0.00024414, 0.,     0.,
 0.,     0.,     0.,     0.,     0.00048828, 0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.00024414, 0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0.,
 0.,     0.,     0.,     0.,     0.,     0. ,
 0.,     0.,     0.,     0.,
]


def maxabs(a, b):
    return np.max(np.abs(a-b))

def test_wavelets_class():
    """tests for wavelets decomposition
    comparisons with modified band and recomp+decomp+extract"""

    print("tests for wavelets decomposition"
    "comparisons with modified band and recomp+decomp+extract")
    img = pywt.data.camera().astype(float)
    level = 3
    detail_level = 2
    band_name = "h"
    wt_inst = Wavelets(img, level, detail_level, band_name)
    wt_inst.decompose()
    # print(wt_inst.dec_obj)

    band = wt_inst.get_band()
    print(band)
    print(band.shape)
    band_modified = band/1.1
    img2 = wt_inst.recompose_from(band_modified)

    wt_inst2 = Wavelets(img2, level, detail_level, band_name)
    wt_inst2.decompose()
    band2 = wt_inst2.get_band()

    # assert np.allclose(img, img2), maxabs(img, img2)
    assert np.allclose(band_modified, band2), maxabs(band_modified, band2)


def test_wavelets_class_recfromzero():
    """tests for wavelets decomposition
    comparisons with modified band and recomp+decomp+extract"""

    print("tests for wavelets decomposition"
    "comparisons with modified band and recomp+decomp+extract")
    img = pywt.data.camera().astype(float)
    level = 3
    detail_level = 1
    band_name = "h"
    wt_inst = Wavelets(img, level, detail_level, band_name)
    wt_inst.decompose()
    # print(wt_inst.dec_obj)

    band = wt_inst.get_band()
    print(band)
    print(band.shape)
    print(wt_inst.get_band(0, 0).shape)
    # band_modified = band/1.1
    img2 = wt_inst.recompose_withnone(band)

    fname_out = "/tmp/recomp_fromzero.png"
    cv2.imwrite(fname_out, img2)

    # wt_inst2 = Wavelets(img2, level, detail_level, band_name)
    # wt_inst2.decompose()
    # band2 = wt_inst2.get_band()

    # assert np.allclose(img, img2), maxabs(img, img2)
    # assert np.allclose(band_modified, band2), maxabs(band_modified, band2)


def test_qim_dwt():
    """tests qim and dwt classes"""
    print("tests qim and dwt classes")
    img_x = pywt.data.camera().astype(float)
    delta = 10
    level = 3
    detail_level = 2
    band_name = "h"

    qim_inst = QIM(delta)
    wt_inst = Wavelets(img_x, level, detail_level, band_name)
    wt_inst.set_params(1, "d")

    wt_inst.decompose()

    band_x = wt_inst.get_band()

    binary_plane = np.random.choice([0, 1], band_x.shape)


    band_y = qim_inst.embed(band_x, binary_plane)
    img_y = wt_inst.recompose_from(band_y)

    v = maxabs(img_x, img_y)
    print("max abs =", v)

    # detection of the inserted binary plane
    wt_inst2 = Wavelets(img_y, level, detail_level, band_name)
    wt_inst2.decompose()
    band_z = wt_inst.get_band()
    band_detected, binary_plane_detected = qim_inst.detect(band_z)

    assert np.allclose(binary_plane, binary_plane_detected)


def test_qimdwt_class():
    """tests qimdwt class"""
    print("tests qimdwt class")
    # img_x = pywt.data.camera().astype(int)

    fname_xs, fname_ts = get_tampered_realistic_png()
    img_nb = 0
    fname_x = EPath(fname_xs[img_nb])
    fname_t = EPath(fname_ts[img_nb])

    small_image = False
    img_x = fname_x.imread_gs_int(small_image=small_image)
    img_t = fname_t.imread_gs_int(small_image=small_image)
    delta = 8
    level = 3
    detail_level = 1
    band_name = "v"



    # deltas = range(16, 60, 4)
    deltas = [10]
    nb_rep = 1

    ber_values = np.zeros(len(deltas))
    i = 0
    for delta in deltas:
        print("delta  = ", delta)
        ber_mean = 0.
        for rep in range(nb_rep):
            detail_level = np.random.choice(range(1, level))
            band_name = np.random.choice(range(3))
            qimdwt_inst = QIMDWT(img_x, delta, level, detail_level, band_name)
            qimdwt_inst.decompose()

            shape = qimdwt_inst.get_band_shape()
            bp = np.random.choice([0, 1], shape)
            qimdwt_inst.embed(bp)
            band_y = qimdwt_inst.get_band()
            img_y = qimdwt_inst.recompose().astype(int)


            mse_val = mtc.compute_mse(img_x, img_y)
            psnr_val = mtc.compute_psnr(img_x, img_y)
            maxabs_val = maxabs(img_x, img_y)

            # print("="*30)
            # print("dlevel = ", detail_level)
            # print("band n = ", band_name)

            # print("rep id = ", rep)
            # print("mse    = ", mse_val)
            print("pnsr   = ", psnr_val)
            # print("maxabs = ", maxabs_val)
            attack_function = image_processings['lpf']
            img_z = img_y
            # img_z = attack_function(img_y, 2)
            img_z, tamper_map = apply_tamper_pattern(img_y, img_x, img_t)

            psnr_val_z = mtc.compute_psnr(img_z, img_y)
            print("pnsr   = ", psnr_val_z)
            cv2.imwrite("/tmp/tamper_map.png", tamper_map)


            # test if the message is still inside the second decomposition
            qimdwt_inst2 = QIMDWT(img_z, delta, level, detail_level, band_name)
            qimdwt_inst2.decompose()


            band_z = qimdwt_inst2.get_band()



            band_detected, bp_dectected = qimdwt_inst2.detect()

            maxabs_val2 = maxabs(bp, bp_dectected)
            ber_val = mtc.compute_ber(bp.flatten(), bp_dectected.flatten())
            ber_mean += ber_val
            # print("maxabs2 = ", maxabs_val2)
            # print("ber     = ", ber_val)

        ber_values[i] = ber_mean/nb_rep
        i += 1
            # assert np.allclose(bp, bp_dectected), ber_val
            # for delta < 7, cannot detect without having errors
            #### strange behavior where the BER is almost 0 or almost 0.9
            #### --> apply rank metric ?? why does that happen ??
            #### bits are reversed when detail_level = 0 : 0.91821
            #### bits are almost identical when detail_level = 0 : 0.0.2392
            #### en fait on n'insere pas souvnet dans la bande 0
    print(ber_values)
    plot_curve(deltas, ber_values)





def plot_curve(X, Y):

    plt.plot(X, Y, color='r', marker='x')
    plt.ylim(-0.01, 1)
    plt.show()



def test_qimdwt_class_rankmetric_errorstructure():
    """tests qimdwt class"""
    print("tests qimdwt class")
    img_x = pywt.data.camera().astype(int)
    delta = 8
    level = 3
    detail_level = 2
    band_name = "h"

    nb_rep = 100
    for delta in range(8, 30):
        for rep in range(nb_rep):

            detail_level = np.random.choice(range(level))
            band_name = np.random.choice(range(3))
            qimdwt_inst = QIMDWT(img_x, delta, level, detail_level, band_name)
            qimdwt_inst.decompose()

            shape = qimdwt_inst.get_band_shape()
            bp = np.random.choice([0, 1], shape)
            qimdwt_inst.embed(bp)
            img_y = qimdwt_inst.recompose().astype(int)


            mse_val = mtc.compute_mse(img_x, img_y)
            psnr_val = mtc.compute_psnr(img_x, img_y)
            maxabs_val = maxabs(img_x, img_y)

            print("="*30)
            print("dlevel = ", detail_level)
            print("band n = ", band_name)
            print("delta  = ", delta)
            print("rep id = ", rep)
            print("mse    = ", mse_val)
            print("pnsr   = ", psnr_val)
            print("maxabs = ", maxabs_val)


            # test if the message is still inside the second decomposition
            qimdwt_inst2 = QIMDWT(img_y, delta, level, detail_level, band_name)
            qimdwt_inst2.decompose()
            band_detected, bp_dectected = qimdwt_inst2.detect()

            maxabs_val2 = maxabs(bp, bp_dectected)
            print("maxabs2 = ", maxabs_val2)
            ber_val = mtc.compute_ber(bp.flatten(), bp_dectected.flatten())
            assert np.allclose(bp, bp_dectected), ber_val
            # for delta < 7, cannot detect without having errors
            #### strange behavior where the BER is almost 0 or almost 0.9
            #### --> apply rank metric ?? why does that happen ??
            #### bits are reversed when detail_level = 0 : 0.91821
            #### bits are almost identical when detail_level = 0 : 0.0.2392


def test_pywt2d_numerically():
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, dtype=np.float64)
    print(x, x.dtype)
    wp = pywt.WaveletPacket2D(data=x, wavelet='db1', mode='symmetric')





def test_pywt2d_withfigures():
    x = pywt.data.camera().astype(np.float32)
    shape = x.shape

    max_lev = 3       # how many levels of decomposition to draw
    label_levels = 3  # how many levels to explicitly label on the plots

    fig, axes = plt.subplots(2, 4, figsize=[14, 8])
    for level in range(0, max_lev + 1):
        if level == 0:
            # show the original image before decomposition
            axes[0, 0].set_axis_off()
            axes[1, 0].imshow(x, cmap=plt.cm.gray)
            axes[1, 0].set_title('Image')
            axes[1, 0].set_axis_off()
            continue

        # plot subband boundaries of a standard DWT basis
        draw_2d_wp_basis(shape, wavedec2_keys(level), ax=axes[0, level],
                         label_levels=label_levels)
        axes[0, level].set_title('{} level\ndecomposition'.format(level))

        # compute the 2D DWT
        c = pywt.wavedec2(x, 'db2', mode='periodization', level=level)
        # normalize each coefficient array independently for better visibility
        c[0] /= np.abs(c[0]).max()
        for detail_level in range(level):
            c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]
        # show the normalized coefficients
        arr, slices = pywt.coeffs_to_array(c)
        axes[1, level].imshow(arr, cmap=plt.cm.gray)
        axes[1, level].set_title('Coefficients\n({} level)'.format(level))
        axes[1, level].set_axis_off()

    plt.tight_layout()
    plt.show()


def test_pywt2d():
    x = pywt.data.camera().astype(np.float32)
    shape = x.shape
    level = 3       # how many levels of decomposition to draw

    fig, axes = plt.subplots(1, 2, figsize=[14, 8])



    # compute the 2D DWT
    c = list(pywt.wavedec2(x, 'db2', mode='periodization', level=level))

    # normalize each coefficient array independently for better visibility
    c[0] /= np.abs(c[0]).max()
    for detail_level in range(level):
        print(len(c[detail_level + 1]))
        c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]

    arr, slices = pywt.coeffs_to_array(c)




    axes[0].imshow(arr, cmap=plt.cm.gray)


    c2 = c
    print(c2[1][0].shape)
    c2[1][0] *= 0
    c2[2][0] *= 0
    c2[3][0] *= 0
    arr2, slices2 = pywt.coeffs_to_array(c2)
    # arr2 = arr
    axes[1].imshow(arr2, cmap=plt.cm.gray)

    plt.show()





def main():
    # test_pywt2d()
    # test_pywt2d()
    # test_wavelets_class()
    # test_qim_dwt()
    # test_qimdwt_class()
    test_wavelets_class_recfromzero()

if __name__ == "__main__":
    sys.exit(main())