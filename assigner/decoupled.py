from functools import partial

import numpy as np

from assigner.base import BaseAssigner, _dist


class RandomAssigner(BaseAssigner):
    """docstring for RandomAssigner"""

    def __init__(self, lookup_dict, seed):
        super(RandomAssigner, self).__init__(lookup_dict)
        self.seed = seed
        np.random.seed(self.seed)

    def assign(self, task, state=None):
        print(task)
        candidates = self.lookup_dict[task]
        i = np.random.choice(len(candidates))
        chosen = candidates[i]
        del self.lookup_dict[task][i]
        return chosen


class DistanceBasedAssigner(BaseAssigner):
    """docstring for DistanceBasedAssigner"""

    def __init__(self, lookup_dict, dist_metric_fn):
        super(DistanceBasedAssigner, self).__init__(lookup_dict)
        self.dist_metric_fn = dist_metric_fn

    def assign(self, task, state=None):
        candidates = self.lookup_dict[task]
        chosen = self.dist_metric_fn(candidates)
        i = candidates.index(chosen)
        del self.lookup_dict[task][i]
        return chosen


CloserPortFirstAssigner = partial(
    DistanceBasedAssigner, dist_metric_fn=partial(min, key=_dist))
FartherPortFirstAssigner = partial(
    DistanceBasedAssigner, dist_metric_fn=partial(max, key=_dist))
