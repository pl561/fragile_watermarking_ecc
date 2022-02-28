from __future__ import print_function
import os
HOME = os.environ["HOME"]
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw


class ImageReader:
    """image reader where the shape is fixed
       more features to be implemented later
       useful for debug mode"""
    def __init__(self, shape=None):
        self.shape = shape

    def imread_gs_int(self, fname):
        img = cv2.imread(fname, 0).astype(int)
        if self.shape is None:
            return img
        else:
            w, h = self.shape
            return img[:w, :h]


def imread_gs_int(fname, small_image=False,
                  pos=(0, 0), shape=(256, 256)):
    """loads an image np array, if smallimage is True, returns a small
    portion of the image"""
    img = cv2.imread(fname, 0).astype(int)
    if small_image:
        w0, h0 = pos
        sw, sh = shape
        img = img[w0:w0+sw, h0:h0+sh]

    return img



def imshow(name, img):
    img = img.astype(np.uint8)
    key = cv2.imshow(name, img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def stack_images(shape, *images):
    """stack images as nbrow x nbcol"""
    images2 = []
    for img in images:
        if len(img.shape) == 2:
            images2.append(gs2color(img))
        else:
            images2.append(img)

    rows = []
    nbrow, nbcol = shape
    for irow in range(nbrow):
        row = images2[irow*nbcol: (irow+1)*nbcol]
        rows.append(row)

    hstacked = [np.hstack(img) for img in rows]
    vstacked = np.vstack(hstacked)

    return vstacked


def gs2color(gs_arr):
    return np.dstack((gs_arr,)*3)


def text2image(text, shape, bgcolor=(255, 255, 255), fgcolor=(0, 0, 0)):
    """writes a text into an image"""
    text_image_result = np.dstack((np.zeros(shape),)*3) + 255.
    min_sh = np.min(shape)
    shape_sq = min_sh, min_sh
    img = Image.new('RGB', shape_sq, color=bgcolor)

    d = ImageDraw.Draw(img)
    d.text((10, 10), text, fill=fgcolor)

    text_image = pil2nparray(img)
    text_image_result[: min_sh, : min_sh] = text_image
    return text_image_result


def pil2nparray(img):
    print(img.size)
    w, h = img.size
    return np.array(img.getdata(), np.uint8).reshape(w, h, 3)


def test_text2image():

    text2 = """
Code                        fwecc.py
Version                          v10
File name                    DPP0029
Image size                  720x1280
p                                  2
s                                 16
C(n, k)                      (16, 1)
embedding                     QIMDWT
auth mode                     BCHDEC
qim delta                         30
dwt level                          2
dwt dlevel                         0
dwt band                          hh
dwt type                        haar
dwt mode               periodization
t_embed                     0.436762
t_tamper                  0.00650311
t_auth                        3.7113
nb cw                              0
nb cw error                    1e-06
nb cw controlled                   0
control rate                       0
control rate on error              0
PSNR                          38.765
MSE                          8.64139
TP                             58649
FP                             12025
FN                              2215
TN                            848711
F1 score                    0.891742
Accuracy                    0.984549
TPR (recall)                0.963607
MCC                         0.886333
PPV (prec)                  0.829853
FPR (false alarm)          0.0139706
FNR (miss detect)          0.0363926
FDR (false disc.)           0.170147
FOR (false omis.)         0.00260305
    """
    # img = Image.new('RGB', (100, 30), color = (73, 109, 137))
    #
    # d = ImageDraw.Draw(img)
    # d.text((10,10), "Hello World", fill=(255,255,0))
    #
    # img.save('/tmp/pil_text.png')
    #
    # return
    text = "PSNR = 40dB\ndelta = 30\nhuhhuu"
    shape1 = 720, 720
    shape2 = 720, 1280
    img = text2image(text2, shape2)
    print(img.shape)
    cv2.imwrite("/tmp/text.png", img)


def main():
    test_text2image()


if __name__ == "__main__":
    sys.exit(main())