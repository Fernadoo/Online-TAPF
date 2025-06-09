import random
from functools import partial

import numpy as np

from assigner.base import BaseAssigner, _dist


class AlphaAssigner(BaseAssigner):
    """docstring for AlphaAssigner"""

    def __init__(self, lookup_dict, alpha):
        super(AlphaAssigner, self).__init__(lookup_dict)
        self.get_closer = partial(min, key=_dist)
        self.get_farther = partial(max, key=_dist)
        self.get_random = partial(min, key=lambda x: random.random())

        self.alpha = alpha

        self.ratio_hist = []

    def assign(self, task, state):

        locations = state['locations']
        locations = np.array(locations)
        max_x, max_y = locations.max(axis=0)
        occup_matrix = np.zeros((max_x + 1, max_y + 1), dtype=int)
        occup_matrix[[locations[:, 0]], locations[:, 1]] = 1

        lhalf, rhalf = (np.mean(occup_matrix[1:, 1: (max_y + 1) // 2]),
                        np.mean(occup_matrix[1:, (max_y + 1) // 2: max_y + 1]))
        self.ratio_hist.append((lhalf, rhalf))

        candidates = self.lookup_dict[task]
        if rhalf > self.alpha:  # best for rule-based touring: 0.235
            chosen = self.get_farther(candidates)
        else:
            chosen = self.get_closer(candidates)
        i = candidates.index(chosen)
        del self.lookup_dict[task][i]
        return chosen
