from __future__ import print_function
import os
HOME = os.environ["HOME"]

import sys
import time
import pandas as pd


def print_time(msg, start, end, unit="ms"):
    diff = end-start
    if unit == "ms":
        diff *= 1000
    elif unit == "s":
        pass
    else:
        unit = ""
    print("{} : {:.3f} {}".format(msg, diff, unit))

class TimeTicker:
    def __init__(self):
        self.t0 = time.time()
        self.t = self.t0

        self.ticks = [self.t0]
        self.last_delta = 0

        self.max_ticks = 100
        self.bookmarks = pd.DataFrame()

    def tick(self, msg=None):
        """computes the current tick"""
        self.t = time.time()
        self.last_delta = self.t - self.t0
        self.t0 = self.t
        self._add_tick()
        self.bookmark(msg)
        return self.last_delta

    def bookmark(self, msg):
        if msg is not None:
            self.bookmarks[msg] = self.t

    def _add_tick(self):
        """adds the newest tick in the tick list"""
        if len(self.ticks) >= self.max_ticks - 1:
            self.ticks = self.ticks[1:]
        self.ticks.append(self.t)

    def print_time(self, msg, elapsed_from=1, unit="ms"):
        """print the time elapsed from the last delta is elapsed_from=1,
        of the time delta from the last last tick if elapsed_from=2, etc"""
        start = self.ticks[-1-elapsed_from]
        end = self.t
        print_time(msg, start, end, unit=unit)




def main():
    ticker = TimeTicker()
    os.system("sleep 2")
    ticker.tick()
    ticker.tick()
    ticker.print_time("Embedding time            ", 1)
    ticker.print_time("Embedding time            ", 2)


if __name__ == "__main__":
    sys.exit(main())