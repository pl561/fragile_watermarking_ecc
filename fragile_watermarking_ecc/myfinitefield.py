import os
HOME = os.environ["HOME"]

import sys
import numpy as np
from sage.all import *


class MyFiniteField(object):
    """finite field with vector space on base field"""
    def __init__(self, q, m):
        self.q, self.m = q, m
        self.field = GF(self.q ** self.m, 'a')
        self.br = self.field.base_ring()
        self.vs = VectorSpace(self.field.base_ring(), self.m)
        self.cardinality = self.field.order()
        self.multiplicative_order = self.cardinality - 1
        self.ff2vs = {}
        self.vs2ff = {}
        self.create_dict()

    def create_dict(self):
        for elt in self.field:
            self.ff2vs[elt] = self.vs(elt)
            self.vs2ff[tuple(self.vs(elt))] = elt

    def tovec(self, ffelt):
        return self.ff2vs[ffelt]

    def toff(self, vec):
        return self.vs2ff[tuple(vec)]

    def random_element(self, repr="ff", nonzero=False):
        elt = self.field.random_element()
        if nonzero:
            while elt == self.field.zero():
                elt = self.field.random_element()

        if repr == "ff":
            return elt
        elif repr == "vs":
            return self.tovec(elt)
        else:
            raise ValueError("unknown representation")

    def random_ffvector(self, vs=None, dim=None, weight=None):
        if vs is None:
            vs = VectorSpace(self.field, dim)
        else:
            if dim is None:
                dim = vs.dimension()
            else:
                raise ValueError("only set one arg between vs and dim")

        if weight is None:
            return vs.random_element()
        else:
            ffvec = [vs(0)] * dim
            indices = np.random.choice(range(dim), weight, replace=False)
            for i in indices:
                ffvec[i] = self.random_element(nonzero=True)
        return vs(ffvec)

    def primitive_element(self):
        return self.field.primitive_element()

    def frobenius_power(self, ffelt, k):
        return ffelt ** (Mod(self.q, self.multiplicative_order) ** k)

    def map_frobenius_power(self, elts, k):
        return [self.frobenius_power(elt, k) for elt in elts]


def test_randomffvector():
    mff = MyFiniteField(2, 8)
    vs = VectorSpace(mff.field, 4)
    for i in range(10):
        elt = mff.random_ffvector(vs=vs, weight=2)

        print(elt)

def main():
    pass


if __name__ == "__main__":
    sys.exit(main())