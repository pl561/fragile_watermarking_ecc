from __future__ import print_function
import os
HOME = os.environ["HOME"]

import sys
import numpy as np
from sage.all import *
from termcolor import colored



def print_vec(msg, vec, sep1=" ", sep2="  ", step=4,
              endline=False, nocolor=False):
    write = sys.stdout.write
    write(msg)
    for index, elt in enumerate(vec):
        if step != 0 and index > 0:
            if index % step == 0:
                write(sep2)
            else:
                write(sep1)
        # elt = elt
        if not nocolor:
            if elt != 0:
                text = colored("{}".format(elt), "red")
            else:
                text = colored("{}".format(elt), "white")
        else:
            text = str(elt)
        write(text)
    if endline:
        write("\n")

def print_matrix(mat, sep=' ', zero="0", one="1"):
    write = sys.stdout.write
    for indw in range(mat.nrows()):
        write("[")
        for indh in range(mat.ncols()):

            elt = mat[indw, indh]
            if elt != 0:
                text = colored(one, "red")
            else:
                text = colored(zero, "white")
            write(text)
            write(sep)
        write("]\n")


def main():
    pass

if __name__ == "__main__":
    sys.exit(main())