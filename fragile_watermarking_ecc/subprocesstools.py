#! /usr/bin/env python
# -*-coding:utf-8 -*

from __future__ import print_function
import sys
import os
from subprocess import Popen, PIPE
import numpy as np
from termcolor import colored
import cv2
HOME = os.environ['HOME']
import delegator


# https://www.pythonforbeginners.com/os/subprocess-for-system-administrators

# http://queirozf.com/entries/python-3-subprocess-examples

# https://github.com/kennethreitz/delegator.py


def do():
    p1 = Popen(['ls', '-la'], stdout=PIPE, stderr=PIPE)
    p_ignore = Popen(['sleep', '4']) # non bloquant, on continue d'executer
    print("hello")


    out, err = p1.communicate()
    print(out)
    print(err)

    print("attente du sleep 4")
    print(p_ignore.communicate())
    print("fin")


def test_delegator():


    commands = []
    for i in range(4):
        c = delegator.run('sleep {}'.format(i+2), block=False)
        commands.append(c)

        print(c.pid)

    c0 = commands[0]

    # print(c0, c0.is_alive)
    # c0.kill()
    # del commands[0]
    # del c0
    # print(c0, c0.is_alive)


    print()
    for c in commands:
        print(c.pid, c.is_alive, c.out)



    # c.block()


    # print(c.kill())



def main():
    # do()
    test_delegator()


if __name__ == '__main__':
    sys.exit(main())