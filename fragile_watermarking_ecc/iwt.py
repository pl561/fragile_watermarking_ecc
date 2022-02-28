from __future__ import print_function
import sys

import numpy as np


def _iwt(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in xrange(ny):
        output[0:x,j] = (array[0::2,j] + array[1::2,j])//2
        output[x:nx,j] = array[0::2,j] - array[1::2,j]
    return output

def _iiwt(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    for j in xrange(ny):
        output[0::2,j] = array[0:x,j] + (array[x:nx,j] + 1)//2
        output[1::2,j] = output[0::2,j] - array[x:nx,j]
    return output

def iwt2(array):
    return _iwt(_iwt(array.astype(int)).T).T

def iiwt2(array):
    return _iiwt(_iiwt(array.astype(int).T).T)





def do():
    shape = 10, 10
    img = np.random.randint(0, 256, shape)
    print(img)
    img_iwt = iwt2(img)
    print(img_iwt)
    img_iiwt = iiwt2(img_iwt)
    print(img_iiwt)
    print(np.allclose(img, img_iiwt))

def do2():
    import pywt
    l = 2
    # x = [3, 7, 1, 1, -2, 5, 4, 6, 8, 5, 7, 2, 6, 8]
    x = np.random.randint(0, 256, 32)
    x = np.array(x)
    print("initial signal :".rjust(20), x)
    wtfilter = pywt.Wavelet('db4')

    coeffs = list(pywt.wavedec(x, wtfilter, level=l))

    print()
    for band in coeffs:
        print("band :".rjust(20), np.array(band))


    ## band modification
    b = coeffs[-1]
    b[2] = 0
    coeffs[-1] = b
    ## replace band


    print()
    rec = pywt.waverec(coeffs, wtfilter)
    print("reconstructed :".rjust(20), rec)
    print("same signal".rjust(20), np.allclose(x, rec))
    diff = np.abs(rec-x)
    print("diff".rjust(20), diff)


def main():
    np.set_printoptions(suppress=True, precision=4)
    do2()



if __name__ == "__main__":
    sys.exit(main())