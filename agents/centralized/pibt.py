"""
Adapted from the official PIBT implementation
See https://github.com/Kei18/pypibt/tree/main
"""

import random
from collections import namedtuple
from queue import PriorityQueue

import numpy as np

from agents.centralized import LiBiaoRobot
from marp.mapf import r_move
from marp.utils import r_Manhattan, Marker


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
        self.NIL_DIR = -1
        self.occupied_now = np.full(self.layout.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(self.layout.shape, self.NIL, dtype=int)
        self.rng = np.random.default_rng(0)
        self.timer = 0

    def act(self, state):
        self.timer += 1
        # if self.timer % 20 >= 0:
        #     self.alg_hist.append('rules')
        #     return super().act(state)

        locations = state['locations']
        directions = state['directions']
        goals = state['goals']
        next_goals = state['next_goals']

        self.curr_goals = []
        for i, ag in enumerate(self.agents):
            self.curr_goals.append(goals[i][next_goals[ag]])

        if getattr(self, 'priorities', None) is None:
            self.priorities = []
            for i, ag in enumerate(self.agents):
                self.priorities.append(
                    r_Manhattan(locations[i], directions[i], self.curr_goals[i]) / self.layout.size
                )
        # print(self.priorities)

        # prioritize occupied agents over idle ones
        idles = self.get_idle_agents(state)
        for i, ag in enumerate(self.agents):
            if i in idles:
                self.priorities[i] -= np.floor(self.priorities[i])

        print(self.priorities[30], self.priorities[20])

        action_n = self.pibtSTEP(locations, directions, self.priorities)

        return action_n

        safe_action_n = self.cand2safe(state, action_n)
        self.alg_hist.append('pibt' + '-unsafe' * (safe_action_n != action_n))
        return safe_action_n

    def get_idle_agents(self, state):
        """
        Given a system state, identify the agents who are idle,
        then assign it a task
        """
        next_goals = state['next_goals']
        idles = []  # (i, agent, loc, drct)
        for i, agent in enumerate(self.agents):
            if next_goals[agent] % 2 == 0:
                idles.append(i)
        return idles

    def pibtSTEP(self, Q_from, D_from, priorities):
        # setup
        N = len(Q_from)
        Q_to = []
        D_to = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            D_to.append(self.NIL_DIR)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        print(A)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                ret = self.pibtFUNC(Q_from, D_from, Q_to, D_to, i)
                if not ret:
                    raise RuntimeError('No successor locations!')
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

        print(Q_to[30], D_to[30], self.curr_goals[30])
        print(Q_to[20], D_to[20], self.curr_goals[20])

        action_n = dict()
        for i, ag in enumerate(self.agents):
            for a in range(4):
                if (Q_to[i], D_to[i]) == r_move((Q_from[i], D_from[i]), a):
                    action_n[ag] = a
                    break

        return action_n

    def pibtFUNC(self, Q_from, D_from, Q_to, D_to, i):
        # true -> valid, false -> invalid

        def get_neighbors(layout, loc, drct):
            nrows, ncols = layout.shape
            neigh = []
            for a in range(4):
                succ_loc, succ_drct = r_move((loc, drct), a)
                if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols):
                    continue
                if layout[succ_loc] in Marker.INACCESSIBLE:
                    continue
                neigh.append((succ_loc, succ_drct))
            return neigh

        # get candidate next vertices, including the current one
        C = get_neighbors(self.layout, Q_from[i], D_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: r_Manhattan(u[0], u[1], self.curr_goals[i]))
        # print(C, Q_from[i], D_from[i], self.curr_goals[i])
        # print(C, [r_Manhattan(u[0], u[1], self.curr_goals[i]) for u in C])
        # exit()

        # vertex assignment
        for (v, d) in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            D_to[i] = d
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and j != i
                and (Q_to[j] == self.NIL_COORD)
                and (D_to[j] == self.NIL_DIR)
                and (not self.pibtFUNC(Q_from, D_from, Q_to, D_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        D_to[i] = D_from[i]
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
