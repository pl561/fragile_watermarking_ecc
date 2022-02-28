from __future__ import print_function
import sys
import os
HOME = os.environ["HOME"]

import numpy as np

# QIM with dfferent quantization steps
# eg : a quantization table 8x8 filled with different quantization coefficients


class QIM:
    def __init__(self, delta, l=100):
        self.delta = delta
        self.l = l

    def embed2(self, x, m):
        """x is a vector of values to be quantized individually
           m is a binary vector of bits to be embeded"""
        x = x.astype(float)
        d = self.delta
        y = np.round(x/d) * d + (-1)**(m+1) * d/4.
        return y

    def detect2(self, z):
        embed = self.embed2

        shape = z.shape
        z = z.flatten()

        m_detected = np.zeros_like(z, dtype=float)
        z_detected = np.zeros_like(z, dtype=float)

        z0 = embed(z, 0)
        z1 = embed(z, 1)

        d0 = np.abs(z - z0)
        d1 = np.abs(z - z1)

        gen = zip(range(len(z_detected)), d0, d1)
        for i, dd0, dd1 in gen:
            # print(dd0, dd1)
            if dd0 < dd1:
                m_detected[i] = 0
                z_detected[i] = z0[i]
            else:
                m_detected[i] = 1
                z_detected[i] = z1[i]


        z_detected = z_detected.reshape(shape)
        m_detected = m_detected.reshape(shape)
        return z_detected, m_detected.astype(int)

    def random_msg(self, l):
        return np.random.choice((0, 1), l)

    def embed_kundur(self, x, m):
        """x is a vector of values to be quantized individually
           m is a binary vector of bits to be embedded"""
        x = x.astype(float)
        d = self.delta
        # y = np.round(x/d) * d + (-1)**(m+1) * d/4.
        mask_x = np.zeros_like(x)


        den = self.delta*2**self.l
        mask_x[np.round(x/den).astype(int) % 2 != m] = 1

        snx = np.zeros_like(x)
        snx[x > 0] = -1
        snx[x <= 0] = 1

        y = x + mask_x * snx * self.delta

        return y

    def detect_kundur(self, z):
        den = self.delta*2**self.l
        m_detected = np.zeros_like(z, dtype=float)
        m_detected = np.round(z/den).astype(int) % 2
        return None, m_detected


def test_qim():
    l = 10000
    delta = 8
    qim = QIM(delta)

    while True:
        x = np.random.randint(0, 255, l).astype(float)


        msg = qim.random_msg(l)
        y = qim.embed2(x, msg)
        z_detected, msg_detected = qim.detect2(y)

        print(x)
        print(y)
        print(z_detected)

        print(msg)
        print(msg_detected)
        assert np.allclose(msg, msg_detected)
        assert np.allclose(y, z_detected)


def main():
    test_qim()


if __name__ == "__main__":
    sys.exit(main())