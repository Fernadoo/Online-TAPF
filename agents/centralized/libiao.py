from copy import deepcopy
from itertools import product

import numpy as np

from marp.mapf import check_collision, r_move
from marp.utils import Manhattan, Marker


class LiBiaoRobot(object):
    """
    Libiao Baseline Robot, based on handcraft rules:

    1. The lanes at the top and at the bottom are delivery lanes,
        while the two middle ones are `venous` lanes.

        1.1 A robot will take a move only if the move is safe
            (under the current rule).

    2. While the battery is below 100 (about a round-tour),
        go to the charging station.

        2.1 A robot on her way to the charging station will be associated
            with the highest priority.

    3. The rule-based the system is a sequence of rules of the form:
            {revised_action_n <- rule_prN_vM(action_n, state)}
        where N means the priority and M for version control.
        Every rule itself should be consistent, i.e. collision-free and lock-free.
    """

    def __init__(self, warehouse):
        # env config
        self.warehouse = warehouse
        self.layout = warehouse.world.layout
        self.charging_stations = warehouse.world.stations
        self.pickup_ports = np.array(np.where(warehouse.world.layout == Marker.IMPORT)).T
        self.delivery_ports = np.array(np.where(warehouse.world.layout == Marker.EXPORT)).T
        self.turnings = np.array(np.where(warehouse.world.layout == Marker.TURNING)).T

        # robots positions
        self.agents = self.warehouse.agents
        self.locations = self.warehouse.get_state()['locations']

        # rule_pr0
        self.delivery_lane = list(product([1], range(2, 83))) + list(product([4], range(2, 85)))
        self.venous_lane = list(product([2], range(2, 83))) + list(product([3], range(2, 85)))

        # rule_pr1
        self.turnings = [tuple(t) for t in self.turnings]
        # print(self.turnings)

        # rule_pr2: cycle
        self.deadlock = False
        self.stuck = False
        self.count_down = 0

    def act(self, state):
        candidate_action_n = self.rule_pr0_v0(state)
        self.cycle_detection(state) #TODO should take in the next planed direction
        if self.stuck and self.count_down > 0:
            self.count_down -= 1
        else:
            self.stuck = False
            self.count_down = 0
            candidate_action_n = self.rule_pr1_v0(state, candidate_action_n)
        # TODO: rule_cargo: agents with cargo give way to agents without cargo
        finalized_action_n = self.cand2safe(state, candidate_action_n)
        return finalized_action_n

    def rule_pr0_v0(self, state):
        locations = state['locations']
        directions = state['directions']

        action_n = dict()
        for i, agent in enumerate(self.agents):
            loc_i = locations[i]
            dir_i = directions[i]

            if loc_i in self.delivery_lane:
                action = go_fwd_only_while_facing(0, dir_i)

            elif loc_i in self.venous_lane:
                action = go_fwd_only_while_facing(180, dir_i)

            elif loc_i == (1, 83):
                action = go_fwd_only_while_facing(270, dir_i)

            elif loc_i == (2, 83):
                action = go_fwd_only_while_facing(180, dir_i)

            elif loc_i == (1, 1):
                action = go_fwd_only_while_facing(0, dir_i)

            elif loc_i == (2, 1):
                action = go_fwd_only_while_facing(90, dir_i)

            elif loc_i in [(4, 85), (3, 85), (2, 85)]:
                action = go_fwd_only_while_facing(90, dir_i)

            elif loc_i == (1, 85):
                action = go_fwd_only_while_facing(180, dir_i)

            elif loc_i in [(1, 84), (2, 84)]:
                action = go_fwd_only_while_facing(270, dir_i)

            elif loc_i == (3, 1):
                action = go_fwd_only_while_facing(270, dir_i)

            elif loc_i == (4, 1):
                action = go_fwd_only_while_facing(0, dir_i)

            else:
                action = 0  # stop by default

            action_n[agent] = action

        return action_n

    def rule_pr0_v1(self, state):
        locations = state['locations']
        directions = state['directions']

        action_n = dict()
        for i, agent in enumerate(self.agents):
            loc_i = locations[i]
            dir_i = directions[i]

            if loc_i in self.delivery_lane:
                action = go_fwd_only_while_facing(180, dir_i)

            elif loc_i == (1, 51):
                action = go_fwd_only_while_facing(180, dir_i)

            elif loc_i == (2, 51):
                action = go_fwd_only_while_facing(90, dir_i)

            elif loc_i == (1, 4):
                action = go_fwd_only_while_facing(270, dir_i)

            elif loc_i == (2, 4):
                action = go_fwd_only_while_facing(0, dir_i)

            elif loc_i in [(1, 53), (2, 53), (3, 53)]:
                action = go_fwd_only_while_facing(270, dir_i)

            elif loc_i == (4, 53):
                action = go_fwd_only_while_facing(180, dir_i)

            elif loc_i == (1, 52):
                action = go_fwd_only_while_facing(0, dir_i)

            elif loc_i in [(2, 52), (3, 52)]:
                action = go_fwd_only_while_facing(90, dir_i)

            elif loc_i == (3, 4):
                action = go_fwd_only_while_facing(0, dir_i)

            elif loc_i == (4, 4):
                action = go_fwd_only_while_facing(90, dir_i)

            elif loc_i in self.venous_lane:
                action = go_fwd_only_while_facing(0, dir_i)

            else:
                action = 0  # stop by default

            action_n[agent] = action

        return action_n

    def rule_pr1_v0(self, state, default_action_n):
        locations = state['locations']
        directions = state['directions']
        goals = state['goals']
        next_goals = state['next_goals']

        # candidate actions
        revised_action_n = deepcopy(default_action_n)
        revised_agents = []
        for i, agent in enumerate(self.agents):
            loc_i = locations[i]
            dir_i = directions[i]
            goal_i = goals[i][next_goals[agent]]

            # if goal_i in [(1, 51), (1, 52)]:
            #     continue

            action = default_action_n[agent]
            if (loc_i in self.turnings) and (loc_i[1] <= goal_i[1]):
                revised_agents.append(agent)
                if (goal_i[0] == 1 and goal_i[1] < 82) or goal_i == (1, 83):
                    action = go_fwd_only_while_facing(90, dir_i)
                elif (goal_i[0] == 4 and goal_i[1] < 82) or goal_i == (1, 84):
                    action = go_fwd_only_while_facing(270, dir_i)

            revised_action_n[agent] = action

        # conflict resolution: in favor of larger ids
        # TODO: might be a 3-agent case: the middle one could do one more redundant step
        for ag_i in revised_agents:
            i = eval(ag_i.split('_')[-1])
            loc_i = locations[i]
            dir_i = directions[i]
            goal_i = goals[i][next_goals[ag_i]]
            for ag_j in revised_agents:
                j = eval(ag_j.split('_')[-1])
                loc_j = locations[j]
                dir_j = directions[j]
                goal_j = goals[j][next_goals[ag_j]]
                if i >= j:
                    continue
                if loc_i[1] != loc_j[1]:
                    continue
                if ((goal_i[0] == 4 and goal_i[1] < 82)
                        and (goal_j[0] == 1 and goal_j[1] < 82)
                        and (loc_i[0] < loc_j[0])) or\
                        ((goal_i[0] == 1 and goal_i[1] < 82)
                         and (goal_j[0] == 4 and goal_j[1] < 82)
                         and (loc_i[0] > loc_j[0])):
                    revised_action_n[ag_i] = default_action_n[ag_i]

        # conflict resolution: in favor of higher priority
        for ag_i in revised_agents:
            i = eval(ag_i.split('_')[-1])
            loc_i = locations[i]
            dir_i = directions[i]
            for j, ag_j in enumerate(self.agents):
                loc_j = locations[j]
                dir_j = directions[j]
                if i == j:
                    continue
                if Manhattan(loc_i, loc_j) > 2:
                    continue
                succ_loc_i, succ_dir_i = r_move((loc_i, dir_i), revised_action_n[ag_i])
                succ_loc_j, succ_dir_j = r_move((loc_j, dir_j), default_action_n[ag_j])
                _, c_j = check_collision((locations[i], locations[j]), (succ_loc_i, succ_loc_j))
                if c_j and default_action_n[ag_j] == 1:  # can rotate in place
                    revised_action_n[ag_j] = 0

        return revised_action_n

    def rule_pr1_v1(self, state, default_action_n):
        locations = state['locations']
        directions = state['directions']
        goals = state['goals']
        next_goals = state['next_goals']

        # candidate actions
        revised_action_n = deepcopy(default_action_n)
        revised_agents = []
        for i, agent in enumerate(self.agents):
            loc_i = locations[i]
            dir_i = directions[i]
            goal_i = goals[i][next_goals[agent]]

            # if goal_i in [(1, 51), (1, 52)]:
            #     continue

            action = default_action_n[agent]
            if loc_i in self.turnings:
                revised_agents.append(agent)
                if goal_i == (1, 51) and loc_i[0] == 1:
                    action = go_fwd_only_while_facing(270, dir_i)
                elif goal_i == (1, 52) and loc_i[0] == 4:
                    action = go_fwd_only_while_facing(90, dir_i)
                elif (goal_i[0] == 4 and goal_i[1] < 50) and loc_i[0] != 4 and loc_i[1] >= goal_i[1]:
                    action = go_fwd_only_while_facing(270, dir_i)
                elif (goal_i[0] == 1 and goal_i[1] < 50) and loc_i[0] != 1 and loc_i[1] >= goal_i[1]:
                    action = go_fwd_only_while_facing(90, dir_i)
            revised_action_n[agent] = action
        # print(revised_agents)

        # conflict resolution: in favor of larger ids
        # TODO: might be a 3-agent case: the middle one could do one more redundant step
        for ag_i in revised_agents:
            i = eval(ag_i.split('_')[-1])
            loc_i = locations[i]
            dir_i = directions[i]
            goal_i = goals[i][next_goals[ag_i]]
            for ag_j in revised_agents:
                j = eval(ag_j.split('_')[-1])
                loc_j = locations[j]
                dir_j = directions[j]
                goal_j = goals[j][next_goals[ag_j]]
                if i >= j:
                    continue
                if loc_i[1] != loc_j[1]:
                    continue
                if ((goal_i[0] == 4 and goal_i[1] < 50)
                        and (goal_j[0] == 1 and goal_j[1] < 50)
                        and (loc_i[0] < loc_j[0])) or\
                        ((goal_i[0] == 1 and goal_i[1] < 50)
                         and (goal_j[0] == 4 and goal_j[1] < 50)
                         and (loc_i[0] > loc_j[0])):
                    revised_action_n[ag_i] = default_action_n[ag_i]

        # conflict resolution: in favor of higher priority
        for ag_i in revised_agents:
            i = eval(ag_i.split('_')[-1])
            loc_i = locations[i]
            dir_i = directions[i]
            for j, ag_j in enumerate(self.agents):
                loc_j = locations[j]
                dir_j = directions[j]
                if i == j:
                    continue
                if Manhattan(loc_i, loc_j) > 2:
                    continue
                succ_loc_i, succ_dir_i = r_move((loc_i, dir_i), revised_action_n[ag_i])
                succ_loc_j, succ_dir_j = r_move((loc_j, dir_j), default_action_n[ag_j])
                _, c_j = check_collision((locations[i], locations[j]), (succ_loc_i, succ_loc_j))
                if c_j and default_action_n[ag_j] == 1:  # can rotate in place
                    revised_action_n[ag_j] = 0

        return revised_action_n

    def cycle_detection(self, state):
        # cycle detection and breaking
        locations = state['locations']
        directions = state['directions']
        # print(f"Locations: {locations}")
        # print(f"Directions: {directions}")

        def adj(loc, direction):
            if direction == 0:
                return tuple(np.add(loc, [0, 1]))
            elif direction == 90:
                return tuple(np.add(loc, [-1, 0]))
            elif direction == 180:
                return tuple(np.add(loc, [0, -1]))
            elif direction == 270:
                return tuple(np.add(loc, [1, 0]))

        visited = []
        cycle = None

        for i, loc_i in enumerate(locations):
            # print(len(visited), len(locations))
            visiting_locs = []
            visiting_id = []
            if loc_i in visited:
                continue

            visited.append(loc_i)
            visiting_locs.append(loc_i)
            visiting_id.append(i)
            curr_loc = loc_i
            curr_dir = directions[i]
            while True:
                next_loc = adj(curr_loc, curr_dir)
                if next_loc not in locations:
                    break
                next_id = locations.index(next_loc)
                if next_loc in visiting_locs:
                    visiting_locs.append(next_loc)
                    visiting_id.append(next_id)
                    cycle = (visiting_id, visiting_locs)
                    break
                next_dir = directions[locations.index(next_loc)]
                visiting_locs.append(next_loc)
                visiting_id.append(next_id)
                visited.append(next_loc)

                curr_loc = next_loc
                curr_dir = next_dir

            if cycle and not ((1, 83) in cycle[1] and (1, 84) in cycle[1]):
                break
            else:
                cycle = None

        if cycle:
            start_i = cycle[1].index(cycle[1][-1])
            breakout = None
            for i in range(start_i, len(cycle[1])):
                loc = cycle[1][i]
                for d in [0, 90, 180, 270]:
                    adj_loc = adj(loc, d)
                    if adj_loc not in locations and self.layout[adj_loc] in Marker.ACCESSIBLE:
                        breakout = (cycle[0][i], d)
                        break
                if breakout:
                    break
            if self.deadlock:  # the second time when the cycle is detected
                self.stuck = True
                self.count_down = 5
                print(f"Victim detection: {cycle}")
                print(f"Cycle found: {cycle[0][start_i:]}")
                if breakout:
                    print(f"To break: agent {breakout[0]} go {breakout[1]}")
                else:
                    print(f"Cannot break cycle")
            else:
                self.deadlock = True
        else:
            self.deadlock = False

    def cand2safe(self, state, action_n):
        finalized_action_n = deepcopy(action_n)
        locations = state['locations']
        directions = state['directions']
        for i, ag_i in enumerate(self.agents):
            safe = True
            for j, ag_j in enumerate(self.agents):
                if i == j:
                    continue
                if Manhattan(locations[i], locations[j]) > 2:
                    continue
                succ_loc_i, succ_dir_i = r_move((locations[i], directions[i]), action_n[ag_i])
                for action_j in [0, action_n[ag_j]]:  # only two possibilities: success or unsuccess
                    succ_loc_j, succ_dir_j = r_move((locations[j], directions[j]), action_j)
                    c_i, _ = check_collision((locations[i], locations[j]), (succ_loc_i, succ_loc_j))
                    if c_i:
                        # print(ag_i, action_n[ag_i], ag_j, action_j)
                        safe = False
                        break
                if not safe:
                    break
            if not safe and action_n[ag_i] == 1:  # can rotate in place
                finalized_action_n[ag_i] = 0

        return finalized_action_n


def go_fwd_only_while_facing(force_dir, dir_i):
    if force_dir == 0:
        if dir_i == 0:
            action = 1  # fwd
        elif dir_i == 90:
            action = 3  # turn right
        elif dir_i == 180:
            action = 3  # turn right
        elif dir_i == 270:
            action = 2  # turn left
    elif force_dir == 90:
        if dir_i == 90:
            action = 1  # fwd
        elif dir_i == 0:
            action = 2  # turn left
        elif dir_i == 180:
            action = 3  # turn right
        elif dir_i == 270:
            action = 3  # turn right
    elif force_dir == 180:
        if dir_i == 180:
            action = 1  # fwd
        elif dir_i == 0:
            action = 2  # turn left
        elif dir_i == 90:
            action = 2  # turn left
        elif dir_i == 270:
            action = 3  # turn right
    elif force_dir == 270:
        if dir_i == 270:
            action = 1  # fwd
        elif dir_i == 0:
            action = 3  # turn right
        elif dir_i == 90:
            action = 3  # turn right
        elif dir_i == 180:
            action = 2  # turn left
    return action
