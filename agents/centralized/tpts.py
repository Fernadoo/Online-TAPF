import random
from collections import namedtuple
from queue import PriorityQueue

import numpy as np

from agents.centralized import LiBiaoRobot
from marp.mapf import r_move
from marp.utils import Manhattan, Marker, r_Manhattan


class TPTSRobot(LiBiaoRobot):
    """docstring for TPTSRobot: Token Passing with Task Swaps"""

    def __init__(self, warehouse, heu):
        super().__init__(warehouse)
        """
        Note that previously,
            self.pickup_ports = np.array(np.where(warehouse.world.layout == Marker.IMPORT)).T
        But IMPORT/EXPORT is inaccessible, the cells next to IMPORT/EXPORT markers are the real ones
        """
        self.pickups = []
        for p in self.pickup_ports:
            for dxdy in [[-1, 0], [0, 1], [1, 0], [0, -1]]:
                cell = tuple(np.add(p, dxdy))
                if cell[0] in range(self.layout.shape[0]) and\
                        cell[1] in range(self.layout.shape[1]) and\
                        self.layout[cell] in Marker.ACCESSIBLE:
                    self.pickups.append(cell)

        # a set of collision-free paths [ [ (prevA, (loc, drct)), ... ], ... ]
        self.token = [[] for _ in range(len(self.agents))]
        self.reserveTab = np.zeros(shape=(*self.layout.shape, 1000), dtype=int) - 1

        self.timer = 0

        self.heu = heu

        # when any agent finds no way out
        self.nowayout = False
        self.nowayout = 0
        self.force_replan = False
        self.alg_hist = []

    def update_reserveTab(self, i, path_i):
        """
        Update reservation table for agent i's path_i
        """
        Tmax = len(path_i)
        for t in range(Tmax):
        # for t in range(min(5, Tmax)):
            prevAction, (loc, drct) = path_i[t]
            self.reserveTab[(*loc, self.timer + t)] = i  # the location is reserved for agent-i in step-t
            # note that t is a relative timestamp onwards

    def act(self, state):
        if self.nowayout:
            action_n = False
        else:
            action_n = self.tpts(state, force_replan=self.force_replan)
            if not action_n:
                self.nowayout = True
                self.nowayout_cntdown = 5  # TODO: should it be larger than stuck_cntdown?

        while not action_n:
            # print(action_n)
            if self.nowayout and self.nowayout_cntdown > 0:
                action_n = super().act(state)
                self.nowayout_cntdown -= 1

                if self.nowayout_cntdown == 0:
                    self.nowayout = False
                    action_n = self.tpts(state, force_replan=True)
                    if not action_n:
                        self.nowayout = True
                        self.nowayout_cntdown = 5

        if not self.nowayout:
            safe_action_n = self.cand2safe(state, action_n)
            if safe_action_n != action_n:
                self.force_replan = True
            else:
                self.force_replan = False

        if self.nowayout:
            alg = 'rules'
        else:
            alg = 'tpts' + '-unsafe' * self.force_replan

        self.timer += 1
        self.alg_hist.append(alg)

        if self.nowayout:
            return action_n
        else:
            return safe_action_n

    def tpts(self, state, force_replan=False):
        """
        Prioritized planning with task preemption
        """
        if force_replan:
            self.reserveTab[:, :, self.timer:] = -1

        action_n = dict()

        agents_need_replan = []
        for j, ag_j in enumerate(self.agents):
            if self.timer + 1 >= len(self.token[j]) or force_replan:
                agents_need_replan.append(j)
            else:
                action_n[ag_j] = self.token[j][self.timer + 1][0]

        locations = state['locations']
        directions = state['directions']
        goals = state['goals']
        next_goals = state['next_goals']

        for i in agents_need_replan:
            # print(i)
            ag_i = self.agents[i]
            curr_goal = goals[i][next_goals[ag_i]]
            path_i = spatial_tempo_Astar(self.layout,
                                         (locations[i], directions[i]),
                                         (curr_goal, None),
                                         self.reserveTab,
                                         self.timer,
                                         self.heu)
            # print(path_i)
            # print(goals[i])

            if not path_i:  # no way out for agent i
                return False
            # if len(path_i) == 1:  # already at pickup, but needs to wait for one more step
            #     # TODO might be problematic:
            #     # reserve table ensures no collision until t0, but not sure about t1
            #     # if it simply adds something in t1, it might collide with others in t1
            #     #    - as they only see collisions up to t0
            #     path_i.append((0, (locations[i], directions[i])))

            # print(len(self.token[i]), self.timer, len(path_i))
            if not self.token[i]:
                # self.token[i] += path_i
                self.token[i] += ['Undef'] * max(0, self.timer + 1 - len(self.token[i]))\
                    + path_i[1:]
            else:
                # print(len(self.token[i][:self.timer + 1]), len(path_i[1:]))
                self.token[i] = self.token[i][:self.timer + 1]\
                    + ['Undef'] * max(0, self.timer + 1 - len(self.token[i]))\
                    + path_i[1:]
            self.update_reserveTab(i, path_i)
            # print(self.token[i])

            # print(i, len(self.token[i]), self.timer, len(path_i))
            action_n[ag_i] = self.token[i][self.timer + 1][0]

        return action_n

    def get_idle_agents(self, state):
        """
        Given a system state, identify the agents who are idle,
        then assign it a task
        """
        locations = state['locations']
        directions = state['directions']
        next_goals = state['next_goals']
        idles = []  # (i, agent, loc, drct)
        for i, agent in enumerate(self.agents):
            if next_goals[agent] % 2 == 0:
                idles.append((i, agent, locations[i], directions[i]))
        return idles


def spatial_tempo_Astar(layout, init, goal, reserveTab, timer, heu):
    """
    Spatial temporal A star search,
    init: (loc, dir), goal: (goal_loc, any_dir=None)
    reservation table: size(layout[0], layout[1], T),
        - if [i, j, t] = True, means loc(i, j) is unavailable at step t
    """
    Node = namedtuple('ANode',
                      ['fValue', 'gValue', 'PrevAction', 'T', 'State'])

    def transit(state, a):
        succ_loc, succ_drct = r_move(state, a)
        nrows, ncols = layout.shape
        if succ_loc[0] not in range(nrows) or succ_loc[1] not in range(ncols):
            return None, 99999
        elif layout[succ_loc] in Marker.INACCESSIBLE:
            return None, 99999
        return (succ_loc, succ_drct), 1

    def heuristic(state):
        cloc, cdir = state
        gloc, _ = goal
        if heu == 'r':
            hval = r_Manhattan(cloc, cdir, gloc)
        else:
            hval = Manhattan(cloc, gloc)
        return hval

    def is_reserved(curr_state, succ_state, curr_t, reserveTab):
        _, _, Tmax = reserveTab.shape
        # time exceeded
        if curr_t + 1 >= Tmax:
        # if curr_t + 1 >= min(5, Tmax):
            return False
        # vertex conflict
        if reserveTab[succ_state[0][0], succ_state[0][1], curr_t + 1] > -1:
            return True
        # edge conflict
        if reserveTab[succ_state[0][0], succ_state[0][1], curr_t] > -1:
            ag_j = reserveTab[succ_state[0][0], succ_state[0][1], curr_t]
            if reserveTab[curr_state[0][0], curr_state[0][1], curr_t + 1] == ag_j:
                return True
        return False

    def get_successors(node):
        f, negG, prev_action, t, curr_state = node
        g = - negG
        successors = []
        for a in range(4):  # 0: stop, 1: fwd, 2: tl, 3: tr
            succ_state, cost = transit(curr_state, a)
            if cost > 9999:  # invalid succ_state
                continue
            if is_reserved(curr_state, succ_state, t, reserveTab):
                continue
            h = heuristic(succ_state)
            tie_break_noise = random.uniform(0, 1e-3)
            succ_node = Node(h + g + cost, -(g + cost + tie_break_noise), a, t + 1, succ_state)
            successors.append(succ_node)
        return successors

    plan = []
    visited = []
    parent_dict = dict()
    q = PriorityQueue()
    q.put(Node(heuristic(init), 0, None, timer, init))

    while not q.empty():  # TODO: init==goal, do we need a stop or nothing?
        curr_node = q.get()

        if curr_node.State[0] == goal[0]:  # as long as locations match
            plan.insert(0, (0, goal))  # do one more `stop` at the goal, might cause collision
            # backtrack to get the plan
            curr = curr_node
            while curr.State != init:
                plan.insert(0, (curr.PrevAction, curr.State))
                curr = parent_dict[str(curr)]
            plan.insert(0, (curr.PrevAction, curr.State))
            return plan

        if str(curr_node.State) in visited:
            continue

        successors = get_successors(curr_node)
        for succ_node in successors:
            q.put(succ_node)
            parent_dict[str(succ_node)] = curr_node
        visited.append(str(curr_node.State))

        if len(visited) > 1 and goal[0] == (-1, -1):
            break

    # if no path, then retrieve the first one in the visited because they are safe
    # can be possible that nothing is visited?
    if len(visited) == 1:
        return False
        # raise RuntimeError("No way out!")
    succ_loc, succ_drct = eval(visited[1])
    for a in range(4):  # 0: stop, 1: fwd, 2: tl, 3: tr
        cand_state, _ = transit(init, a)
        if (succ_loc, succ_drct) == cand_state:
            plan.insert(0, (a, (succ_loc, succ_drct)))
            break
    plan.insert(0, (None, init))
    # print(plan)
    return plan
    # raise RuntimeError("No astar plan found!")
