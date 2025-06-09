import os
import re
from copy import deepcopy

import numpy as np

from agents.centralized import LiBiaoRobot, go_fwd_only_while_facing
from marp.mapf import check_collision, r_move
from marp.utils import Manhattan, Marker

ALGOLIB = {
    "eecbs": "/Users/fernando/Documents/EECBS/eecbs",
}
TMP_PREFIX = 'tmp'


class CBSRobot(LiBiaoRobot):
    """
    General CBSRobot, supporting cbs, eecbs, rhcr, pibt2, e.t.c.
    """

    def write_layout_file(self, layout_file=f"{TMP_PREFIX}/tmp.map"):
        def marker2char(marker):
            if marker in Marker.ACCESSIBLE:
                return '.'
            else:
                return '@'

        with open(layout_file, 'w') as lf:
            lf.write(f"type warehouse\n")
            lf.write(f"height {self.layout.shape[0]}\n")
            lf.write(f"width {self.layout.shape[1]}\n")
            lf.write(f"map\n")
            layout = []
            for i in range(self.layout.shape[0]):
                row = ''.join(list(map(marker2char, self.layout[i]))) + '\n'
                layout.append(row)
            lf.writelines(layout)

        self.layout_file = layout_file
        return layout_file

    def write_agent_file(self, locations, agent_file=f"{TMP_PREFIX}/tmp.scen"):
        with open(agent_file, 'w') as af:
            af.write(f"version 1\n")
            alines = []
            valid_locations = []
            valid_goals = []
            valid_agents = []
            for i, ag in enumerate(self.agents):
                x, y = locations[i]
                g_x, g_y = self.warehouse.world.goals[i][self.warehouse.world.next_goals[ag]]
                if g_x not in range(self.layout.shape[0]) or g_y not in range(self.layout.shape[1]):
                    continue
                if (g_x, g_y) in valid_goals:
                    # continue
                    if (x, y) in valid_goals:
                        continue
                    line = f"0\ttmp_map\ttmp_h\ttmp_w\t{y}\t{x}\t{y}\t{x}\ttmp_opt\n"
                else:
                    line = f"0\ttmp_map\ttmp_h\ttmp_w\t{y}\t{x}\t{g_y}\t{g_x}\ttmp_opt\n"
                alines.append(line)
                valid_locations.append((x, y))
                valid_goals.append((g_x, g_y))
                valid_agents.append(ag)
            af.writelines(alines)

        self.agent_file = agent_file
        return agent_file, valid_agents

    def call_solver(self, locations, alg='eecbs', sol_file=f"{TMP_PREFIX}/paths.txt"):
        solver_path = ALGOLIB[alg]

        if alg == 'eecbs':
            if getattr(self, 'layout_file', None) is None:
                layout_file = self.write_layout_file()
            else:
                layout_file = self.layout_file

            agent_file, valid_agents = self.write_agent_file(locations)

            timeout = 0.1
            subopt = 1.5
            args = [
                f"{solver_path}",
                f"-m {layout_file}",
                f"-a {agent_file}",
                f"-k {len(valid_agents)}",
                f"--outputPaths={sol_file}",
                f"-t {timeout}",
                f"--suboptimality={subopt}",
            ]
            # TODO: a more elegant way via os.subprocess
            cmd = " ".join(args)
            os.system(f"{cmd} > {TMP_PREFIX}/sol.log")

            # print(re.split(",| ", open(f"{TMP_PREFIX}/sol.log", 'r').read().split(": ")[1]))
            if re.split(",| ", open(f"{TMP_PREFIX}/sol.log", 'r').read().split(": ")[1])[0] != 'Succeed':
                # exit()
                return None

        return sol_file, valid_agents

    def read_sol(self, sol_file, valid_agents):
        paths = {}
        with open(sol_file, 'r') as sf:
            line = sf.readline()
            while line:
                # e.g., Agent 0: (16,5)->(17,5)->(17,6)->
                chunks = re.split(': |->', line)
                ag_i = eval(chunks[0].split()[-1])
                paths[valid_agents[ag_i]] = list(map(lambda s: eval(s), chunks[1: -1]))
                line = sf.readline()
        return paths

    def act(self, state):
        candidate_action_n = self.rule_pr0_v0(state)
        candidate_action_n = self.cbs_actions_pr1(state, candidate_action_n)
        candidate_action_n = self.rule_pr1_v0(state, candidate_action_n)
        finalized_action_n = self.cand2safe(state, candidate_action_n)
        return finalized_action_n

    def cbs_actions_pr1(self, state, default_action_n):
        locations = state['locations']
        directions = state['directions']

        # TODO: consider one step lookahead?
        sol_ret = self.call_solver(locations)
        if sol_ret:
            print('by cbs')
            sol_file, valid_agents = sol_ret
            paths = self.read_sol(sol_file, valid_agents)
        else:
            return default_action_n

        revised_action_n = deepcopy(default_action_n)
        revised_agents = []
        for i, agent in enumerate(self.agents):
            if agent in paths and len(paths[agent]) > 1:
                curr_loc = np.array(locations[i])
                next_loc = np.array(paths[agent][1], dtype=int)
                # print(curr_loc, next_loc)
                dir_to_move = dxdy2dir(tuple(next_loc - curr_loc))
                # print(dir_to_move)
                if dir_to_move == 'stop':
                    next_action = 0
                else:
                    next_action = go_fwd_only_while_facing(dir_to_move, directions[i])
                revised_action_n[agent] = next_action
                revised_agents.append(agent)

        # collision resolution in favor of higher priority (cbs actions)
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
