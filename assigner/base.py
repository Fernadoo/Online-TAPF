from copy import deepcopy


class BaseAssigner(object):
    """docstring for BaseAssigner"""

    def __init__(self, lookup_dict):
        super(BaseAssigner, self).__init__()
        self.lookup_dict = deepcopy(lookup_dict)

    def assign(self, task, state=None):
        # given a task and the current state, select one of the ports according to the lookup_dict
        pass


def _dist(x):
    return (x <= 74) * abs(x - 74) + (x >= 75) * abs(x - 75)
