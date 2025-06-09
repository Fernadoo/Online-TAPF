"""
Adapted from the official PIBT implementation
See https://github.com/Kei18/pypibt/tree/main
"""

import random
from collections import namedtuple
from queue import PriorityQueue

import numpy as np

from agents.centralized import LiBiaoRobot, go_fwd_only_while_facing
from marp.mapf import move
from marp.utils import Manhattan, Marker


class PIBTRobot(LiBiaoRobot):
    """docstring for PIBTRobot: priority inheritance with backtracking"""

    def __init__(self, warehouse):
        super().__init__(warehouse)
        """
        Note that previously,
            self.pickup_ports = np.array(np.where(warehouse.world.layout == Marker.IMPORT)).T
        But IMPORT/EXPORT is inaccessible, the cells next to IMPORT/EXPORT markers are the real ones
        """
        # for hist tracking
        self.alg_hist = []

        # for priority inheritance
        self.priorities = None
        self.NIL = -1
        self.NIL_COORD = (-1, -1)
        self.occupied_now = np.full(self.layout.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(self.layout.shape, self.NIL, dtype=int)
        self.rng = np.random.default_rng(0)

    def act(self, state):
        locations = state['locations']
        goals = state['goals']
        next_goals = state['next_goals']

        self.directions = state['directions']
        self.curr_goals = []
        for i, ag in enumerate(self.agents):
            self.curr_goals.append(goals[i][next_goals[ag]])

        if getattr(self, 'priorities', None) is None:
            self.priorities = []
            for i, ag in enumerate(self.agents):
                self.priorities.append(Manhattan(locations[i], self.curr_goals[i]) / self.layout.size)

        action_n = self.pibtSTEP(locations, self.priorities)
        safe_action_n = self.cand2safe(state, action_n)
        self.alg_hist.append('pibt' + '-unsafe' * (safe_action_n != action_n))
        return safe_action_n

    def pibtSTEP(self, Q_from, priorities):
        # setup
        N = len(Q_from)
        Q_to = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.pibtFUNC(Q_from, Q_to, i)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        # update priorities
        for i in range(N):
            if Q_to[i] != self.curr_goals[i]:
                priorities[i] += 1
            else:
                priorities[i] -= np.floor(priorities[i])

        action_n = dict()
        for i, ag in enumerate(self.agents):
            curr_loc = np.array(Q_from[i], dtype=int)
            succ_loc = np.array(Q_to[i], dtype=int)
            dir_to_move = dxdy2dir(tuple(succ_loc - curr_loc))
            if dir_to_move == 'stop':
                next_action = 0
            else:
                next_action = go_fwd_only_while_facing(dir_to_move, self.directions[i])
            action_n[ag] = next_action

        return action_n

    def pibtFUNC(self, Q_from, Q_to, i):
        # true -> valid, false -> invalid

        def get_neighbors(layout, loc):
            nrows, ncols = layout.shape
            neigh = []
            for a in range(5):
                succ_loc = move(loc, a)
                if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols):
                    continue
                if layout[succ_loc] in Marker.INACCESSIBLE:
                    continue
                neigh.append(succ_loc)
            return neigh

        # get candidate next vertices, including the current one
        C = get_neighbors(self.layout, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: Manhattan(u, self.curr_goals[i]))

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.pibtFUNC(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False


def dxdy2dir(dxdy):
    if dxdy == (0, 0):
        return 'stop'  # stop
    elif dxdy == (-1, 0):
        return 90  # up
    elif dxdy == (0, 1):
        return 0  # right
    elif dxdy == (1, 0):
        return 270  # down
    elif dxdy == (0, -1):
        return 180  # left
