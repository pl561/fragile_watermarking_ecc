import os
HOME = os.environ["HOME"]

import sys
import numpy as np

import cv2
import arrayprocessing as ap
import pandas as pd


def compute_ber(s0, s1):
    """computes the binary error rate of two binary sequences s0 and s1"""
    positions = np.where(s0 != s1)[0]
    ber_value = len(positions)/float(len(s0))
    return ber_value

def compute_mse(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    result = (a-b)**2
    h, w = result.shape
    return result.sum()/h/w

def compute_psnr(im1, im2):
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    MAX = 255
    res_mse = compute_mse(im1, im2)
    if res_mse == 0:
        res = np.inf
    else:
        res = 20 * np.log10(MAX) - 10 * np.log10(res_mse)
    return res

def confusion_matrix(tm1, tm2):
    """returns the confusion matrix from two tamper maps
       tm1 is the state, tm2 is the prediction"""
    # assert 255 in tm1 and 255 in tm2
    # assert 0 in tm1 and 0 in tm2
    tp = len(np.where((tm1 == 255) & (tm2 == 255))[0])
    fp = len(np.where((tm1 == 0) & (tm2 == 255))[0])
    fn = len(np.where((tm1 == 255) & (tm2 == 0))[0])
    tn = len(np.where((tm1 == 0) & (tm2 == 0))[0])

    conf_mat = np.array([[tp, fp], [fn, tn]])
    return conf_mat

def confusion_map(tm1, tm2):
    """builds a confusion map (color image) to
       observe both the true tamper map and
       the authentication tamper map"""
    # no other values than 0 and 255 can be inside the np arrays
    # assert 255 in tm1 and 255 in tm2
    # assert 0 in tm1 and 0 in tm2

    h, w = tm1.shape
    shape = h, w, 3
    confusion_map = np.zeros(shape)
    tp_pos = np.where((tm1 == 255) & (tm2 == 255))
    fp_pos = np.where((tm1 == 0) & (tm2 == 255))
    fn_pos = np.where((tm1 == 255) & (tm2 == 0))
    tn_pos = np.where((tm1 == 0) & (tm2 == 0))

    grayscale = 100
    # https://www.rapidtables.com/web/color/orange-color.html
    # b, g, r
    confusion_map[tp_pos] = [0, 255, 0] # real tampered pixels
    # confusion_map[fp_pos] = [0, 69, 255] # detect more than really tampered
    confusion_map[fp_pos] = [255, 0, 0] # detect more than really tampered
    confusion_map[fn_pos] = [0, 0, 255] # missed detection, didn't detect the tampered pixels
    confusion_map[tn_pos] = [grayscale, grayscale, grayscale] # real untampered pixels

    return confusion_map

def autocrop_confusionmap(confmap, border_sz=20):
    """autocrop confusion map"""
    grayscale = 100
    try:
        gray_map = confmap.astype(float).sum(axis=2)/3.
        X, Y = np.where(gray_map < grayscale)
        xmin, xmax = X.min()-border_sz, X.max()+border_sz
        ymin, ymax = Y.min()-border_sz, Y.max()+border_sz
        cropped = confmap[xmin: xmax, ymin: ymax]
    except:
        cropped = confmap
    return cropped


def binarymap(m1, m2):
    """
    return an array indicating in white (255) a pixel difference and in black
    a pixel equality
    :param m1:
    :param m2:
    :return:
    """
    # work in integers
    # m1 = m1.astype(int)
    # m2 = m2.astype(int)
    # assert len(m1[m1 != 0 and m1 != 1][0]) == 0
    # assert len(m1[m2 != 0 and m2 != 1][0]) == 0
    diff = (m1 + m2) % 2
    return diff * 255.


def window_sliding_binarymap(error_map, nb_pixel=4, ws=3):
    """
    a 3x3 window centered at x_i, y_i sliding on the error_map that determines
    the number s of non zero values,
    if s >= nb_pixel then the pixel value at x_i, y_i is marked as 1 or 255 for image
    repr purpose  in the newly generated map ws_error_map
    :param error_map: 2D array with 0 and 1 (or 255)
    :param ws: window size
    :param nb_pixel: the threshold number of non zero values in a window
    :return: ws_error_map
    """
    ws_error_map = np.zeros_like(error_map)
    gen = ap.window_sliding(error_map, ws=ws)
    for w0, w1, h0, h1 in gen:
        b = error_map[w0: w1, h0: h1]
        nb_nonzero_pix = len(np.where(b != 0)[0])
        x_m, y_m = (w0+w1)/2, (h0+h1)/2

        if error_map[x_m, y_m] > 0:
            nb_nonzero_pix -= 1

        if nb_nonzero_pix >= nb_pixel:
            ws_error_map[x_m, y_m] = 1
    return ws_error_map * 255.


def metric_m1(error_map):
    """
    implements metric M1 from Qi et al. 2015
    :param map1: binary map 1
    :param map2: binary map 2
    :return: ratio of nb of white pixels over nb of pixels
    """
    nb_white_pixels = len(np.where(error_map != 0)[0])
    nb_pixels = error_map.shape[0] * error_map.shape[1]
    ratio = nb_white_pixels/float(nb_pixels)
    return ratio


class ConfusionData:
    """
    A class that represents data of the confusion matrix as a confusion map, etc
    It can compute different metrics from experiment analysis
    """
    def __init__(self, truth_map, predicted_map):
        self.truth_map = truth_map
        self.predicted_map = predicted_map
        self.conf_matrix = confusion_matrix(self.truth_map, self.predicted_map)
        self.conf_map = confusion_map(self.truth_map, self.predicted_map)

    def compute_metrics(self):
        cm = self.conf_matrix
        f1 = compute_f1(cm)
        mcc = compute_mcc(cm)
        ppv = compute_ppv(cm)
        fnr = compute_fnr(cm)
        fpr = compute_fpr(cm)
        recall = compute_tpr(cm)

        names_values = [
            ("F1",         f1),
            ("MCC",        mcc),
            ("Prec",       ppv),
            ("False Al.",  fpr),
            ("Miss Det.",  fnr),
            # ("Recall",     recall),
        ]

        df_metrics = pd.DataFrame()
        for n, v in names_values:
            df_metrics[n] = [np.round(v, 4)]

        return df_metrics.T




def compute_f1(confusion_matrix):
    """read this webpage to understand the measures
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient"""
    tp, fp, fn, tn = confusion_matrix.astype(np.float64).flatten()
    a = 2*tp
    f1 = a/(a+fp+fn)
    return f1

def compute_mcc(confusion_matrix):
    tp, fp, fn, tn = confusion_matrix.astype(np.float64).flatten()
    num = tp * tn - fp * fn
    den = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    mcc = num/np.sqrt(den)
    return mcc

def compute_acc(cm):
    """computes accuracy"""
    tp, fp, fn, tn = cm.astype(np.float64).flatten()
    acc = (tp+tn) / (tp+fp+fn+tn)
    return acc

def compute_ppv(cm):
    """computes precision or also called positive predictive value"""
    tp, fp, fn, tn = cm.astype(np.float64).flatten()
    ppv = tp / (tp+fp)
    return ppv

def compute_fnr(cm):
    """computes missed detection rate or false negative rate"""
    tp, fp, fn, tn = cm.astype(np.float64).flatten()
    fnr = fn/(fn+tp)
    return fnr

def compute_fpr(cm):
    """false alarm rate or false positive rate"""
    tp, fp, fn, tn = cm.astype(np.float64).flatten()
    fpr = fp/(fp+tn)
    return fpr

def compute_tpr(cm):
    """recall or true positive rate"""
    tp, fp, fn, tn = cm.astype(np.float64).flatten()
    tpr = tp/(tp+fn)
    return tpr

def compute_fdr(cm):
    """False Discovery Rate"""
    tp, fp, fn, tn = cm.astype(np.float64).flatten()
    fdr = fp/(fp+tp)
    return fdr

def compute_for(cm):
    """False Omission Rate"""
    tp, fp, fn, tn = cm.astype(np.float64).flatten()
    false_omission_rate = fn/(fn+tn)
    return false_omission_rate

# https://stats.stackexchange.com/questions/118219/how-to-interpret-matthews-correlation-coefficient-mcc
# https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
# https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/
# https://en.wikipedia.org/wiki/F1_score
# table in diagnostic testing section of wikipedia page is very good
# How to interpret and critisize F1 score ?

def test_autocrop():
    fname = "/home/plefevre/tmp/fwecc_16_16_LOC/images/DPP0012_x_confmap_loc.png"

    confmap = cv2.imread(fname)
    cropped = autocrop_confusionmap(confmap)
    cv2.imwrite("/tmp/autocrop_test.png", cropped)

def main():
    test_autocrop()


if __name__ == "__main__":
    sys.exit(main())