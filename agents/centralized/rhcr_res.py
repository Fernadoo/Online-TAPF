import random
from collections import namedtuple
from copy import deepcopy
from queue import PriorityQueue

import numpy as np

from agents.centralized import LiBiaoRobot
from marp.mapf import r_move, check_collision
from marp.utils import Manhattan, Marker, r_Manhattan


class CNode():
    """docstring for the Node class in the constraint tree"""

    def __init__(self, layout, locations, directions, goals, reserveTab, ct, horizon, heu):
        self.layout = layout
        self.locations = locations
        self.directions = directions
        self.goals = goals
        self.reserveTab = reserveTab
        self.ct = ct  # a compact repr of reserveTab
        self.horizon = horizon
        self.heu = heu

        self.paths, self.lengths = self.comp_paths(horizon=self.horizon)
        if self.paths is None:
            self.cost = 99999
        else:
            self.cost = sum(self.lengths)
            self.get_collisions(self.paths, self.lengths)

    def comp_paths(self, horizon):
        path = []
        lengths = []
        for i in range(len(self.locations)):
            pi = self.STsearch(
                self.layout,
                (self.locations[i], self.directions[i]),
                (self.goals[i], None),
                self.reserveTab,
                horizon
            )  # path = [(prevAction, (loc, dir)), ... , (0, (gloc, None))]
            if not pi:
                return None, 99999
            path.append(pi)
            lengths.append(len(pi))
        return path, lengths

    def STsearch(self, layout, init, goal, reserveTab, horizon):
        """
        Single-agent spatial temporal A star, dealing with collisions only within `horizon`
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
            if self.heu == 'r':
                hval = r_Manhattan(cloc, cdir, gloc)
            else:
                hval = Manhattan(cloc, gloc)
            return hval

        def is_reserved(curr_state, succ_state, curr_t):
            _, _, Tmax = reserveTab.shape
            # time exceeded
            if curr_t + 1 >= Tmax:
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
                if is_reserved(curr_state, succ_state, t):
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
        q.put(Node(heuristic(init), 0, None, 0, init))

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

        # If no path, then retrieve the first one in the visited because they are safe
        # Is it possible that nothing is visited?
        if len(visited) == 1:
            return False
        succ_loc, succ_drct = eval(visited[1])
        for a in range(4):  # 0: stop, 1: fwd, 2: tl, 3: tr
            cand_state, _ = transit(init, a)
            if (succ_loc, succ_drct) == cand_state:
                plan.insert(0, (a, (succ_loc, succ_drct)))
                break
        plan.insert(0, (None, init))
        return plan

    def get_collisions(self, paths, lengths):
        Lmax = min(max(lengths), self.horizon)
        self.collisions = []

        # paths = [..., pi, pj, pk, ...]
        # pi = [(prevAction, (loc, dir)), ... , (0, (gloc, None))]
        prev_locs = [pi[0][1][0] for pi in paths]
        for t in range(1, Lmax):
            curr_locs = []
            for i in range(len(paths)):
                now = t
                if t >= len(paths[i]):
                    now = len(paths[i]) - 1
                curr_locs.append(paths[i][now][1][0])

            collisions = check_collision(prev_locs, curr_locs)
            for c_i in collisions:
                if c_i:
                    # the first collision happens at t at this location
                    self.collisions = (t, collisions, curr_locs)
                    return collisions

            prev_locs = curr_locs

    def get_children(self):
        if getattr(self, 'collisions', None) is None:
            raise RuntimeError('Search paths first!')
        t, collisions, locs = self.collisions  # collisions = [... , [...], [...], ...]
        # print(collisions)
        children = []
        # print(collisions)
        for i, c_i in enumerate(collisions):
            if c_i:
                for j in [i] + c_i:
                    new_reserveTab = deepcopy(self.reserveTab)
                    new_reserveTab[locs[j] + (t,)] = j
                    # print(f"{j} occupies {locs[j]} at {t}")
                    new_ct = deepcopy(self.ct)
                    new_ct.append((j, locs[j], t))
                    new_cnode = CNode(
                        self.layout, self.locations, self.directions, self.goals, new_reserveTab,
                        ct=new_ct,
                        horizon=self.horizon,
                        heu=self.heu
                    )
                    if new_cnode.cost > 9999:
                        continue
                    children.append(new_cnode)
        return children

    def is_feasible(self):
        if getattr(self, 'collisions', None) is None:
            raise RuntimeError('Search paths first!')
        return len(self.collisions) == 0

    def __lt__(self, other):
        return (self.cost, *self.lengths) < (other.cost, *other.lengths)

    def __repr__(self):
        return str(self.ct)


def CBS(layout, locations, directions, goals, horizon, heu):
    visited = []
    reserveTab = np.zeros(shape=(*layout.shape, horizon + 1), dtype=int) - 1
    q = PriorityQueue()

    q.put(
        CNode(layout, locations, directions, goals, reserveTab,
              ct=[], horizon=horizon, heu=heu)
    )
    # print(f"=== New CBS ===")
    while not q.empty():
        if len(q.queue) > 50:
            print(False, len(q.queue))
            return False

        # print(f"current q len: {len(q.queue)}")
        curr_node = q.get()
        if curr_node.is_feasible():  # no collision within the horizon
            print(True, len(q.queue))
            return curr_node.paths

        if str(curr_node) in visited:
            continue

        # print('<-- conflict splitting..')
        successors = curr_node.get_children()
        # print(f'conflict split into {len(successors)} -->')
        for succ_node in successors:
            q.put(succ_node)
        visited.append(str(curr_node))

    print(False, len(q.queue))
    return False


class RHCRRobot(LiBiaoRobot):
    """docstring for RHCRRobot"""

    def __init__(self, warehouse, horizon=3, heu='r'):
        super().__init__(warehouse)

        self.alg_hist = []
        self.heu = heu
        self.horizon = horizon

        self.nowayout = False
        self.nowayout_cntdown = 0

    def act(self, state):
        locations = state['locations']
        directions = state['directions']
        goals = state['goals']
        next_goals = state['next_goals']

        curr_goals = []
        for i, ag in enumerate(self.agents):
            curr_goals.append(goals[i][next_goals[ag]])

        used_rule = False
        # if self.nowayout_cntdown == 0:
        #     action_n = self.get_actions_from_CBS(locations, directions, curr_goals)
        #     if not action_n:
        #         self.nowayout_cntdown = 5
        #         action_n = super().act(state)
        #         used_rule = True
        # else:
        #     used_rule = True
        #     action_n = super().act(state)
        #     self.nowayout_cntdown -= 1

        action_n = self.get_actions_from_CBS(locations, directions, curr_goals)
        if not action_n:
            action_n = super().act(state)
            used_rule = True

        safe_action_n = self.cand2safe(state, action_n)

        self.alg_hist.append(
            'rules' * used_rule
            + ('rhcr' + '-unsafe' * (safe_action_n != action_n)) * (not used_rule)
        )

        return safe_action_n

    def get_actions_from_CBS(self, locations, directions, curr_goals):
        # paths = [..., pi, pj, pk, ...]
        # pi = [(prevAction, (loc, dir)), ... , (0, (gloc, None))]
        paths = CBS(self.layout, locations, directions, curr_goals, horizon=self.horizon, heu=self.heu)
        if not paths:
            return False

        action_n = dict()
        for i, ag in enumerate(self.agents):
            action_n[ag] = paths[i][1][0]
        return action_n
