#! /usr/bin/env python
# -*-coding:utf-8 -*

import sys
import os

import numpy as np
from termcolor import colored
import cv2

HOME = os.environ['HOME']

# import stackprinter
# stackprinter.set_excepthook(style='color')
# https://github.com/cknd/stackprinter
# https://www.tuicool.com/articles/QjI3eyY

import pysnooper


@pysnooper.snoop()
def do():
    l = "jkl"
    ll = int(l)

def main():
    do()


if __name__ == '__main__':
    sys.exit(main())