import random
from copy import deepcopy

import numpy as np

from marp.mapf import move, get_avai_actions, check_collision
from marp.mapf import r_move, r_get_avai_actions
from marp.mapd import MAPD, MAPD_R
from marp.utils import Marker


STARTS = [(4, 1), (1, 4), (1, 6)]
GOALS = [
    [(6, 6), ],
    [(4, 4), (4, 1), (3, 6), ],
    [(3, 2), (6, 2), ],
]
REWARDS = {
    'illegal': -10000,
    'normal': -1,
    'collision': -1000,
    'goal': 10000
}
BATTERY = 15
CONTINGENCY = 0.0


class Warehouse(MAPD):
    """docstring for Warehouse"""

    def __init__(self, N, layout,
                 starts=STARTS, goals=GOALS, rewards=REWARDS,
                 full_battery=BATTERY, contingency_rate=CONTINGENCY,
                 obs_fn=None, render_mode='human'):
        super().__init__(N, layout,
                         starts, goals, rewards,
                         obs_fn, render_mode)
        self.MAX_NUM_STEP = np.inf
        self.full_battery = full_battery
        self.contingency_rate = contingency_rate
        self.stations = np.array(np.where(self.layout == Marker.BATTERY)).T

    def _reset(self, seed=None, options=None):
        observations, infos = super()._reset(seed, options)
        self.batteries = {agent: self.full_battery for agent in self.agents}
        for agent in self.agents:
            infos[agent]['battery'] = self.full_battery
        self.history = {
            'paths': [self.starts],
            'batteries': [tuple(self.full_battery for agent in self.agents)]
        }
        return observations, infos

    def _step(self, actions):
        succ_locations = []
        if random.random() < self.contingency_rate:
            actions = {
                agent: self._action_space(agent).sample() for agent in self.agents
            }
        rewards = {agent: self.REWARDS['normal'] for agent in self.agents}
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            if self.batteries[agent] <= 0:
                _a = 'stop'
                self.batteries[agent] += 1  # restore
            elif self.terminations[agent]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['goal']
                self.batteries[agent] += 1  # restore
            elif not self.info_n[agent]['action_mask'][_a]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['illegal']
            succ_loc = move(self.locations[i], _a)
            succ_locations.append(succ_loc)
            self.batteries[agent] -= 1

            if self.layout[succ_loc] == Marker.BATTERY:
                self.batteries[agent] = self.full_battery

        collisions = check_collision(self.locations, succ_locations)
        self.locations = succ_locations
        self.history['paths'].append(succ_locations)
        self.history['batteries'].append(tuple(self.batteries[agent] for agent in self.agents))

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            c_i = collisions[i]
            if c_i and not self.terminations[agent]:
                # TODO: incur collision penalty even if the goal is reached in this step
                rewards[agent] = self.REWARDS['collision']
            observations[agent] = self.obs_fn(succ_locations, agent)
            infos[agent] = {
                'action_mask': get_avai_actions(succ_locations[i], self.layout)[1],
                'collide_with': c_i,
                'battery': self.batteries[agent]
            }
        self.obs_n = deepcopy(observations)
        self.info_n = deepcopy(infos)

        for i, agent in enumerate(self.agents):
            if self.locations[i] == self.goals[i][self.next_goals[agent]]:
                rewards[agent] = self.REWARDS['goal']
                if not self.terminations[agent] and self.next_goals[agent] == len(self.goals[i]) - 1:
                    self.terminations[agent] = True
                self.next_goals[agent] = min(self.next_goals[agent] + 1, len(self.goals[i]) - 1)

        terminations = deepcopy(self.terminations)

        self.step_cnt += 1
        if self.step_cnt >= self.MAX_NUM_STEP:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        if np.all(list(terminations.values())) or self.step_cnt >= self.MAX_NUM_STEP:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _render(self):
        if self.render_mode == 'human':
            from marp.animator import WarehouseAnimation
            if max(self.layout.shape) in range(10):
                FPS = 60
            elif max(self.layout.shape) in range(10, 15):
                FPS = 30
            elif max(self.layout.shape) >= 15:
                FPS = 15
            self.animator = WarehouseAnimation(
                range(self.N),
                self.layout,
                self.starts,
                self.goals,
                self.history['paths'],
                self.history['batteries'],
                FPS
            )
            self.animator.show()
        else:
            for step in self.history['paths']:
                print(step)

    def _get_state(self):
        # summarize the information state
        state = {
            'locations': deepcopy(self.locations),
            'infos': deepcopy(self.info_n),
            'goals': deepcopy(self.goals),
            'next_goals': deepcopy(self.next_goals),
            'batteries': deepcopy(self.batteries),
            'stations': deepcopy(self.stations),
        }
        return state

    def _transit(self, state, actions):
        if np.all(self._is_goal_state(state)):
            return state, True

        locations = state['locations']
        infos = state['infos']
        goals = state['goals']
        next_goals = state['next_goals']
        batteries = state['batteries']

        succ_locations = []
        succ_next_goals = deepcopy(next_goals)
        succ_batteries = deepcopy(batteries)
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            succ_batteries[agent] = batteries[agent] - 1
            if not infos[agent]['action_mask'][_a]:
                _a = 'stop'
            succ_loc = move(locations[i], _a)
            if succ_loc == goals[i][next_goals[agent]]:
                succ_next_goals[agent] = min(next_goals[agent] + 1, len(goals[i]) - 1)
            succ_locations.append(move(locations[i], _a))

            if self.layout[succ_loc] == Marker.BATTERY:
                succ_batteries[agent] = self.full_battery

        collision_free = True
        succ_infos = {}
        collisions = check_collision(locations, succ_locations)
        for i, agent in enumerate(self.agents):
            c_i = collisions[i]
            if c_i:
                collision_free = False
            succ_infos[agent] = {
                'action_mask': get_avai_actions(succ_locations[i], self.layout)[1],
                'collide_with': c_i,
            }

        succ_state = {
            'locations': succ_locations,
            'infos': succ_infos,
            'goals': goals,
            'next_goals': succ_next_goals,
            'batteries': succ_batteries,
            'stations': state['stations'],
        }

        return succ_state, collision_free


class Warehouse_R(MAPD_R):
    """docstring for Warehouse_R"""

    def __init__(self, N, layout,
                 starts=STARTS, goals=GOALS, rewards=REWARDS,
                 full_battery=BATTERY, contingency_rate=CONTINGENCY,
                 obs_fn=None, render_mode='human'):
        super().__init__(N, layout,
                         starts, goals, rewards,
                         obs_fn, render_mode)
        self.MAX_NUM_STEP = np.inf
        self.full_battery = full_battery
        self.contingency_rate = contingency_rate
        self.stations = np.array(np.where(self.layout == Marker.BATTERY)).T

    def _reset(self, seed=None, options=None):
        observations, infos = super()._reset(seed, options)
        self.batteries = {agent: self.full_battery for agent in self.agents}
        for agent in self.agents:
            infos[agent]['battery'] = self.full_battery
        self.history['batteries'] = [tuple(self.full_battery for agent in self.agents)]
        return observations, infos

    def _step(self, actions):
        succ_locations = []
        succ_directions = []
        if random.random() < self.contingency_rate:
            actions = {
                agent: self._action_space(agent).sample() for agent in self.agents
            }
        rewards = {agent: self.REWARDS['normal'] for agent in self.agents}
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            if self.batteries[agent] <= 0:
                _a = 'stop'
                self.batteries[agent] += 1  # restore
            elif self.terminations[agent]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['goal']
                self.batteries[agent] += 1  # restore
            elif not self.info_n[agent]['action_mask'][_a]:
                _a = 'stop'
                rewards[agent] = self.REWARDS['illegal']
            succ_loc, succ_drct = r_move((self.locations[i], self.directions[i]), _a)
            succ_locations.append(succ_loc)
            succ_directions.append(succ_drct)
            self.batteries[agent] -= 1

            if self.layout[succ_loc] == Marker.BATTERY:
                self.batteries[agent] = self.full_battery

        collisions = check_collision(self.locations, succ_locations)
        self.locations = succ_locations
        self.directions = succ_directions
        self.history['paths'].append(succ_locations)
        self.history['directions'].append(succ_directions)
        self.history['batteries'].append(tuple(self.batteries[agent] for agent in self.agents))

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            c_i = collisions[i]
            if c_i and not self.terminations[agent]:
                # TODO: incur collision penalty even if the goal is reached in this step
                rewards[agent] = self.REWARDS['collision']
            observations[agent] = self.obs_fn(succ_locations, succ_directions, agent)
            infos[agent] = {
                'action_mask': r_get_avai_actions((succ_locations[i], succ_directions[i]),
                                                  self.layout)[1],
                'collide_with': c_i,
                'battery': self.batteries[agent]
            }
        self.obs_n = deepcopy(observations)
        self.info_n = deepcopy(infos)

        for i, agent in enumerate(self.agents):
            if self.locations[i] == self.goals[i][self.next_goals[agent]]:
                rewards[agent] = self.REWARDS['goal']
                if not self.terminations[agent] and self.next_goals[agent] == len(self.goals[i]) - 1:
                    self.terminations[agent] = True
                self.next_goals[agent] = min(self.next_goals[agent] + 1, len(self.goals[i]) - 1)

        terminations = deepcopy(self.terminations)

        self.step_cnt += 1
        if self.step_cnt >= self.MAX_NUM_STEP:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        if np.all(list(terminations.values())) or self.step_cnt >= self.MAX_NUM_STEP:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _append_new_goals(self, new_goals, verbose=1):
        for agent in new_goals:
            i = eval(agent.split('_')[-1])
            self.goals[i] = self.goals[i] + new_goals[agent]
            self.terminations[agent] = False
            self.next_goals[agent] = min(self.next_goals[agent] + 1, len(self.goals[i]) - 1)
            if verbose:
                print(f"New task for {new_goals}")

    def _render(self):
        if self.render_mode == 'human':
            from marp.animator import WarehouseAnimation_R
            if max(self.layout.shape) in range(10):
                FPS = 60
            elif max(self.layout.shape) in range(10, 15):
                FPS = 30
            elif max(self.layout.shape) >= 15:
                FPS = 15
            elif max(self.layout.shape) >= 50:
                FPS = 8
            self.animator = WarehouseAnimation_R(
                range(self.N),
                self.layout,
                self.starts,
                self.goals,
                self.history['paths'],
                self.history['directions'],
                self.history['batteries'],
                FPS
            )
            self.animator.show()
        else:
            for step in self.history['paths']:
                print(step)

    def _get_state(self):
        # summarize the information state
        state = {
            'locations': deepcopy(self.locations),
            'directions': deepcopy(self.directions),
            'infos': deepcopy(self.info_n),
            'goals': deepcopy(self.goals),
            'next_goals': deepcopy(self.next_goals),
            'batteries': deepcopy(self.batteries),
            'stations': deepcopy(self.stations),
        }
        return state

    def _transit(self, state, actions):
        if np.all(self._is_goal_state(state)):
            return state, True

        locations = state['locations']
        directions = state['directions']
        infos = state['infos']
        goals = state['goals']
        next_goals = state['next_goals']
        batteries = state['batteries']

        succ_locations = []
        succ_directions = []
        succ_next_goals = deepcopy(next_goals)
        succ_batteries = deepcopy(batteries)
        for i, agent in enumerate(self.agents):
            _a = actions[agent]
            succ_batteries[agent] = batteries[agent] - 1
            if not infos[agent]['action_mask'][_a]:
                _a = 'stop'
            succ_loc, succ_drct = r_move((locations[i], directions[i]), _a)
            if succ_loc == goals[i][next_goals[agent]]:
                succ_next_goals[agent] = min(next_goals[agent] + 1, len(goals[i]) - 1)
            succ_locations.append(succ_loc)
            succ_directions.append(succ_drct)

            if self.layout[succ_loc] == Marker.BATTERY:
                succ_batteries[agent] = self.full_battery

        collision_free = True
        succ_infos = {}
        collisions = check_collision(locations, succ_locations)
        for i, agent in enumerate(self.agents):
            c_i = collisions[i]
            if c_i:
                collision_free = False
            succ_infos[agent] = {
                'action_mask': r_get_avai_actions((succ_locations[i], succ_directions[i]),
                                                  self.layout)[1],
                'collide_with': c_i,
            }

        succ_state = {
            'locations': succ_locations,
            'directions': succ_directions,
            'infos': succ_infos,
            'goals': goals,
            'next_goals': succ_next_goals,
            'batteries': succ_batteries,
            'stations': state['stations'],
        }

        return succ_state, collision_free
