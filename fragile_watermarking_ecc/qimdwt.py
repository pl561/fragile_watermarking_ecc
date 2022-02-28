from __future__ import print_function
import os

HOME = os.environ["HOME"]

import sys
import numpy as np
from qim import QIM
from wavelets import Wavelets
from loggingtools import setup_logger


class QIMDWT:
    def __init__(self, img, delta, level,
                 detail_level, band_name,
                 wt_type="db2", mode='periodization', verbose=False):
        """QIM + DWT : objects must be different from embedding and detection
           DWT decomposition and recomposition are independant operations
           from embedding and detection"""
        self.delta = delta
        self.level = level
        self.wt_type = wt_type
        self.mode = 'periodization'

        self.detail_level = detail_level
        self.band_name = band_name

        self.wt_obj = Wavelets(img, level, detail_level, band_name,
                               wt_type=self.wt_type, mode=self.mode)
        self.qim_obj = QIM(self.delta, self.detail_level)


        logging_dir = os.path.join(HOME, "tmp/qimdwt.log")
        s = "QIMDWT"
        self.qimdwt_logger = setup_logger(s, logging_dir)

        if verbose:
            print(self.wt_type)
            print(self.mode)

    def get_band(self, detail_level=None, band_name=None):
        band = self.wt_obj.get_band(detail_level, band_name)
        return band

    def get_band_shape(self, detail_level=None, band_name=None):
        shape = self.wt_obj.get_band(detail_level, band_name).shape
        return shape

    def decompose(self):
        """decompose the image with the wavelet object"""
        self.wt_obj.decompose()

    # def embed_orig(self, binary_plane, detail_level=None, band_name=None):
    #     """embed a mark inside the selected band inside the decomposition """
    #     band_x = self.wt_obj.get_band(detail_level, band_name)
    #     band_y = self.qim_obj.embed(band_x, binary_plane)
    #     self.wt_obj.modify_band(band_y, detail_level, band_name)

    def embed(self, binary_plane, detail_level=None, band_name=None):
        """embed a mark inside the selected band inside the decomposition """
        band_x = self.wt_obj.get_band(detail_level, band_name)
        band_y = self.qim_obj.embed2(band_x, binary_plane)
        self.wt_obj.modify_band(band_y, detail_level, band_name)
        msg = "embed d={} lvl={} dlvl={} bdn={} t={}".format(self.delta,
                                                             self.level,
                                                             self.detail_level,
                                                             self.band_name,
                                                             self.wt_type)
        self.qimdwt_logger.info(msg)

    def recompose(self):
        img_y = self.wt_obj.recompose()
        return img_y

    def recompose_withnone(self, band, detail_level=None, band_name=None):
        return self.wt_obj.recompose_withnone(band, detail_level, band_name)

    # def detect_orig(self, detail_level=None, band_name=None):
    #     band_z = self.wt_obj.get_band(detail_level, band_name)
    #     band_detected, binary_plane_detected = self.qim_obj.detect(band_z)
    #     return band_detected, binary_plane_detected

    def detect(self, detail_level=None, band_name=None):
        band_z = self.wt_obj.get_band(detail_level, band_name)
        band_detected, binary_plane_detected = self.qim_obj.detect2(band_z)
        msg = "detect d={} lvl={} dlvl={} bdn={} t={}".format(self.delta,
                                                              self.level,
                                                              self.detail_level,
                                                              self.band_name,
                                                              self.wt_type)
        self.qimdwt_logger.info(msg)

        return band_detected, binary_plane_detected


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())