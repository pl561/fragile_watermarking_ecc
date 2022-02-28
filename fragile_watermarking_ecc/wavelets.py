from __future__ import print_function
import os

HOME = os.environ["HOME"]

import sys
import numpy as np
import pywt
from loggingtools import setup_logger


# https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html#d-multilevel-reconstruction-using-waverec2
# https://pywavelets.readthedocs.io/en/latest/regression/dwt-idwt.html#


wt_types = []
for f in pywt.families():
    wt_types.extend(pywt.wavelist(f))

wt_modes = pywt.Modes.modes


band_codes = {
    "h"  : 0,
    "lh" : 0,
    "0"  : 0,
    "v"  : 1,
    "hl" : 1,
    "1"  : 1,
    "d"  : 2,
    "hh" : 2,
    "2"  : 2,
}


class Wavelets:
    """implement watermarking embeddings mechanisms using pywt
    a - LL, low-low coefficients
    h - LH, low-high coefficients
    v - HL, high-low coefficients
    d - HH, high-high coefficients
    (cA, (cH, cV, cD))
    https://pywavelets.readthedocs.io/en/latest/regression/wp2d.html
    """
    def __init__(self, image, level,
                 detail_level, band_name,
                 wt_type="db4", mode='periodization'):
        self.image = np.copy(image).astype(float)
        self.level = level

        self.detail_level = detail_level
        self.band_name = band_name

        self.wt_type = wt_type
        self.mode = 'periodization'

        log_file = os.path.join(HOME, "tmp/qimdwt.log")
        s = "DWT   "
        self.dwt_logger = setup_logger(s, log_file)


    def set_params(self, detail_level, band_name):
        if detail_level is not None:
            self.detail_level = detail_level
        if band_name is not None:
            self.band_name = band_name

    def decompose(self):
        self.dec_obj = list(pywt.wavedec2(self.image, self.wt_type,
                                 mode=self.mode, level=self.level))
        # first element must be a numpy array
        for l in range(1, self.level+1):
            self.dec_obj[l] = list(self.dec_obj[l])
        self.arr, self.slices = pywt.coeffs_to_array(self.dec_obj)

    def get_band(self, detail_level=None, band_name=None):
        self.set_params(detail_level, band_name)
        band_id = self.get_band_id(band_name)
        msg = "get band band_id={} dlvl={}".format(band_id, self.detail_level)
        self.dwt_logger.info(msg)
        if self.detail_level != 0:
            self.extracted_band = self.dec_obj[self.detail_level][band_id]
        else:
            self.extracted_band = self.dec_obj[self.detail_level]
        return self.extracted_band

    def modify_band(self, modified_band, detail_level=None, band_name=None):
        self.set_params(detail_level, band_name)
        band_id = self.get_band_id(band_name)
        if self.detail_level != 0:
            # print(self.detail_level, band_id)
            # print(type(self.dec_obj[self.detail_level]))
            # print(type(self.dec_obj[self.detail_level][band_id]))
            self.dec_obj[self.detail_level][band_id] = modified_band
        else:
            self.dec_obj[self.detail_level] = modified_band

    def recompose(self):
        self.modified_image = pywt.waverec2(self.dec_obj, self.wt_type,
                                            mode=self.mode)
        return self.modified_image

    def recompose_from(self, modified_band,
                       detail_level=None, band_name=None):
        self.modify_band(modified_band,
                         detail_level=detail_level, band_name=band_name)
        return self.recompose()


    def recompose_withnone(self, band,
                           detail_level=None, band_name=None):
        self.set_params(detail_level, band_name)

        # t = [None, None, None]
        # dec = [np.zeros_like(band)]+ [t]*(self.level)
        dec = [np.zeros_like(self.dec_obj[0])]
        for dlevel in range(1, len(self.dec_obj)):
            band_zero = np.zeros_like(self.dec_obj[dlevel][0])
            dec.append([band_zero]*3)



        band_id = self.get_band_id(self.band_name)



        if self.detail_level == 0:
           dec[self.detail_level] = band
        else:
            dec[self.detail_level][band_id] = band

        recomposed_withnone = pywt.waverec2(dec, self.wt_type,
                                            mode=self.mode)
        return recomposed_withnone


    def get_band_id(self, band_name):
        """return the band id
        a - LL, low-low coefficients
        h - LH, low-high coefficients
        v - HL, high-low coefficients
        d - HH, high-high coefficients
        (cA, (cH, cV, cD))
        """
        if band_name is None:
            band_name = self.band_name
        s = str(band_name).lower()
        # print(s)
        return band_codes[s]


def test_pywt():
    from pathtools import EPath
    ep_lena = EPath(HOME).join("Images/lena.png")
    img = ep_lena.imread_gs_int()
    dec_obj = list(pywt.wavedec2(img, "haar", level=2))
    print(len(dec_obj))
    for elt in dec_obj:
        if not isinstance(elt, tuple):
            print("not a tuple")
            print(type(elt))
        else:
            print(tuple)
            for subelt in elt:
                print(type(subelt))


def main():
    test_pywt()


if __name__ == "__main__":
    sys.exit(main())
