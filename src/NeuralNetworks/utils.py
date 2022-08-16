""" Summary
"""
import os
import time
# import numpy as np


class ProgressBar:
    """_summary_
    """

    def __init__(self, num_iters, header=None) -> None:
        self._t0 = time.time()
        self._n = num_iters
        self.term_cols = (os.get_terminal_size().columns)*0.15

        if header is not None:
            print(header)
        print(f"[{'.'*int(self.term_cols/self._n)}] loop started..!", end="\r")

    def update(self, i: int) -> None:
        """_summary_

        :param i: _description_
        :type i: int
        """
        tc_points = int(((i+1)/self._n) * self.term_cols)
        avg_time = (time.time()-self._t0)/(i+1)
        percent = ((i+1)/self._n)*100
        #
        completion = f"{percent :04.2f}%"
        remaining_time = f" | TTF: {avg_time*(self._n-(i+1)):04.1f}s"
        bar_str = f"[{'='*(tc_points)+'>'+'.'*int(self.term_cols-tc_points)}]"
        print(f"{bar_str}{completion} {remaining_time}", end="\r")
