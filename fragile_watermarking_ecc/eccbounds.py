from __future__ import print_function
import os
HOME = os.environ["HOME"]

import sys
import numpy as np
from sage.all import *
from termcolor import colored

def binomial_list(a, b, n):
    """computes binomial coeffs binomial(n, k)
       for k = a to k = b"""
    binomials = [binomial(n, i) for i in range(a, b+1)]
    return vector(binomials)

def binomialsum_f(a, b, n, coeff):
    """computes the sum from i=a to i=b binomial(n, i)*coeff^i"""
    binomials = binomial_list(a, b, n)
    coeffs = vector([coeff**i for i in range(a, b+1)])
    ret = binomials*coeffs # scalar product
    return ret

def binomial_sum(a, b, n):
    """computes the sum of binomial coeffs binomial(n, k)
       for k = a to k = b"""
    binomials = binomial_list(a, b, n)
    s = sum(binomials)
    return s

def lower_bound_th22(m, t, l, e, r, q=2):
    """(kumar2019codes)
       Theorem 2.2. An (n, k) linear EL-code over GF (q) which is
       divided into m mutually exclusive sub-blocks of length t each, is
       capable of detecting any l(<= m/2) or less corrupted sub-blocks
       each containing e or less errors."""

    p1 = binomialsum_f(0, e-1, t-1, q-1)
    c = binomialsum_f(1, e, t, q-1)
    p2 = binomialsum_f(0, l-1, m-1, c)
    RHS = p1 * p2

    LHS = q ** r
    cond = LHS > RHS
    return cond


def upper_bound_th22(m, t, l, e, r, q=2):
    """(kumar2019codes)
       Theorem 3.1. An (n, k) linear code over GF (q) which
       is divided into m mutually exclusive sub-blocks of
       length t each, is capable of locating any l(<= m)
       or less corrupted sub-blocks each containing e
       or less errors."""

    # p1 = binomialsum_f()
    #
    # p2 = binomialsum_f()
    RHS = None #p1 * p2
    #
    LHS = q ** r
    cond = LHS > RHS
    return cond


# 3. Location of e or less errors in multiple sub-blocks
# This section is devoted for obtaining lower and upper bounds on the number
# of check digits required for the existence of a linear code over GF (q) capable
# of locating multiple corrupted sub-blocks each containing e or less errors.
# Firstly, we obtain the lower bound on the number of check digits for such
# a linear code and we follow similar approach used in Theorem 1 of [17].

def lower_bound_th31(m, t, l, e, r, q=2):
    """Theorem 3.1. An (n, k) linear code over GF (q) which is divided into m mutually exclusive sub-blocks of length t each, is capable of locating any l(<= m) or less corrupted sub-blocks each containing e or less errors."""

    s1 = binomialsum_f(1, e//2, t, q-1)
    s2 = binomialsum_f(1, l, m, s1)
    RHS = 1 + s2
    #
    LHS = q ** r
    cond = LHS > RHS
    return cond

def upper_bound_th33(m, t, l, e, r, q=2):
    """Theorem 3.3. An (n, k) linear EL-code over GF (q) which is divided into m mutually exclusive sub-blocks of length t each, is capable of locating any l(<= m/2) or less corrupted sub-blocks each containing e or less errors."""

    p1 = binomialsum_f(0, e-1, t-1, q-1)

    s = binomialsum_f(1, e, t, q-1)
    p2 = binomialsum_f(0, 2*l-1, m-1, s)
    RHS = p1 * p2
    #
    LHS = q ** r
    cond = LHS > RHS
    return cond


def lbound_elc_sb(s, t, e):
    emax = int(np.floor(e/2))
    binomials = [binomial(t, i) for i in range(1, emax+1)]
    print(binomials)
    lbound = np.log2(1 + s * sum(binomials))
    return lbound

def test_lbound_elc_sb():
    s, t, e = 4, 4, 2
    s, t, e = map(int, sys.argv[1: 4])
    lbound = lbound_elc_sb(s, t, e)



    print("*"*20)
    print("Block de taille t = {}".format(t))
    print("Nombre de blocs s = {}".format(s))
    print("e-EL code       e = {}".format(e))
    print("*"*20)
    print("r >= {}".format(int(np.ceil(lbound))))
    print("2e = {}".format(2*e))

    cond = (int(np.ceil(lbound)) < 2*e)
    trad = {
        False : "Non",
        True  : "Oui"
    }
    print("Peut-on trouver r <= 2e ?", trad[cond])
    print(" {:.2f} <= r < {}".format(lbound, 2*e))

def test_bound0():
    m = 16
    t = 16
    l = m//2
    # e = t ##//2
    # r = 30
    q = 2
    el_length = m*t

    for e in range(1, t//2+1):
        print("*******************************")
        print("[e = {}]".format(e))
        print("*******************************")
        for r in range(1, el_length):
            cond_lower = lower_bound_th22(m, t, l, e, r, q=q)
            cond_upper = upper_bound_th33(m, t, l, e, r, q=q)
            if cond_upper and cond_lower:
                e_colored = colored("e={}".format(e), "yellow")
                r_colored = colored("r={}".format(r), "red")
                code_str = "EL code of m={} blocks of length t={} over GF({})"
                print(code_str.format(m, t, q))
                print("This code has length n = mt = {}".format(m*t))
                param_str = "Detects at most l={} (<= m={}) " \
                            "blocks containing {} (<= t={}) or fewer errors"
                print(param_str.format(l, m, e_colored, t))

                parity_msg = "Minimum number of " \
                 "parity check digits {}".format(r_colored)
                print(parity_msg)
                print("EL code power ratio : {:.2f}".format(float(l)/m))
                print("RS code power ratio : {:.2f}".format(float(r)/(2*m*t)))
                print("-------------------------------")






def main():
    # test_lbound_elc_sb()
    test_bound0()


if __name__ == "__main__":
    sys.exit(main())
