#! /usr/bin/env python
# -*-coding:utf-8 -*

from __future__ import print_function
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

from pathtools import EPath
from qimdwt import QIMDWT
import metrics as mtc
from wavelets import band_codes



def benchmark():
    data_dir = EPath(HOME).join("tmp/bench_qimdwt")
    data_dir.mkdir()



HOME = os.environ['HOME']


def do():
    pass


def main():
    do()


if __name__ == '__main__':
    sys.exit(main())