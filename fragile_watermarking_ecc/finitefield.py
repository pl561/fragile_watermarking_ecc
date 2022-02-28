import os
HOME = os.environ["HOME"]

import sys
from sage.all import *


def vsgfq(n, q, varname='a'):
    field = GF(q, varname)
    vs = VectorSpace(field, n)
    return vs



def main():
    pass


if __name__ == "__main__":
    sys.exit(main())