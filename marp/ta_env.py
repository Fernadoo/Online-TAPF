import random
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


class MATARP(Env):
    """
    The base MARP environment that unifies the APIs of the supporting tasks

    Args:
        N (int): the number of agents
        layout (str): the file name of the layout configuration
        one_shot (bool): one-shot path finding or lifelong
        render_mode (str or None): will visualize if 'human', otherwise, only print in the console
    """
    metadata = {
        "name": "multiagent_task_assignment_and_route_planning_v0",
    }

    def __init__(self, path_planning_env_fn, path_planner_fn, num_agents,
                 log_file, task_seed, robot_seed,
                 verbose=1,
                 random=False,
                 **kwargs):
        self.path_planning_env_fn = path_planning_env_fn  # currently restricted to warehouse150_1 layout
        self.path_planner_fn = path_planner_fn
        self.num_agents = num_agents

        # extract tasks and task-port correspondence
        log = pd.read_excel(log_file, sheet_name=None)
        self.lookup_table, self.delivery_records = log.values()

        lookup_dict = dict()
        for rid, row in self.lookup_table.iterrows():
            if row['商品编号'] not in lookup_dict:
                lookup_dict[row['商品编号']] = []
            for _ in range(row['计划量']):
                lookup_dict[row['商品编号']].append(row['格口'])
        self.lookup_dict_backup = deepcopy(lookup_dict)

        # random seeds for reproducible experiments
        self.task_seed = task_seed
        self.robot_seed = robot_seed
        self.random = random

        self.verbose = verbose

    def reset(self, seed=None, options=None):
        """
        Reset tasks
        """
        self.tasks = np.array((self.delivery_records['商品编号']))
        self.lookup_dict = deepcopy(self.lookup_dict_backup)

        if self.random:
            if seed is None:
                task_seed, robot_seed = self.task_seed, self.robot_seed
            else:
                task_seed, robot_seed = seed * 618, seed * 1998
        else:
            task_seed, robot_seed = self.task_seed, self.robot_seed

        np.random.seed(task_seed)
        np.random.shuffle(self.tasks)

        random.seed(robot_seed)
        possible_starts = list(product(range(1, 5), range(1, 82)))
        starts = random.sample(possible_starts, k=self.num_agents)

        self.possible_goals = list(product([1], range(8, 82))) + list(product([4], range(81, 5, -1)))

        # print(self.tasks[:20])
        # for t in self.tasks[:5]:
        #     print(list(map(lambda t: self.possible_goals[t - 1], self.lookup_dict[t])))
        # exit()

        goals = []
        for i, start in enumerate(starts):
            if start[0] in [1, 2]:
                goals.append([(1, 83)])
            elif start[0] in [3, 4]:
                goals.append([(1, 84)])
        kwargs = {
            'starts': starts,
            'goals': goals,
            'full_battery': np.inf
        }
        self.path_planning_env = self.path_planning_env_fn(
            N=self.num_agents, layout='warehouse150_1',
            orthogonal_actions=False, one_shot=False, battery=True, render_mode='human',
            **kwargs)

        self.path_planning_env.reset()
        # self.path_planning_env.render()
        # exit()
        self.path_planner = self.path_planner_fn(self.path_planning_env)

        self.curr_t = 0
        self.delivered = 0
        self.finished = 0
        self.it = 0
        while self.path_planning_env.agents:
            robot_state = self.path_planning_env.get_state()
            robot_actions = self.path_planner.act(robot_state)
            self.path_planning_env.step(robot_actions)
            self.it += 1

            if self.verbose:
                print(f"Currently assigned tasks: {min(self.curr_t, len(self.tasks))}")

            # will stop for one step at the previous pickup: robot_state is one-step delayed
            reached_goal_n = self.path_planning_env.is_goal_state(robot_state)
            self.agents_reached_goal = np.where(reached_goal_n)[0].tolist()
            if self.agents_reached_goal:
                break

        obs = (self.tasks[self.curr_t],
               self.lookup_dict[self.tasks[self.curr_t]],
               self.path_planning_env.get_state())
        info = {
            'tasks': deepcopy(self.tasks),
            'lookup_dict': deepcopy(self.lookup_dict),
            'layout': self.path_planning_env.world.layout
        }

        return obs, info

    def step(self, action):
        """
        An action is actually a new task assignment
        for one of the agents who has achieved her current task
        """
        assert action in range(1, 151)
        assert action in self.lookup_dict[self.tasks[self.curr_t]]

        # assign the selected non_trivial task to an idle agent
        if self.agents_reached_goal and self.curr_t < len(self.tasks):

            new_goal_id = action
            new_goal = self.possible_goals[new_goal_id - 1]
            i = self.lookup_dict[self.tasks[self.curr_t]].index(new_goal_id)
            del self.lookup_dict[self.tasks[self.curr_t]][i]
            self.curr_t += 1

            if new_goal[0] == 1:
                pickup = (1, 83)
                new_tasks = [new_goal, pickup]
                self.delivered += 1
            elif new_goal[0] == 4:
                pickup = (1, 84)
                new_tasks = [new_goal, pickup]
                self.delivered += 1

            ag_i = self.agents_reached_goal[0]
            self.path_planning_env.world._append_new_goals({f'robot_{ag_i}': new_tasks},
                                                           verbose=self.verbose)
            self.agents_reached_goal = self.agents_reached_goal[1:]  # assume agent selection in order

        term = False
        trunc = False
        T = self.it
        # R_base = 1
        # R_task = 10
        R = 0
        # gamma = 0.99
        while not (term or trunc):
            # 1. all non-trivial tasks are accomplished
            if self.agents_reached_goal and self.curr_t >= len(self.tasks):
                for i, ag_i in enumerate(self.agents_reached_goal):
                    new_task = [(-1, -1)]  # for dummy destination
                    self.finished += 1
                    self.path_planning_env.world._append_new_goals({f'robot_{ag_i}': new_task},
                                                                   verbose=self.verbose)
                    self.agents_reached_goal = self.agents_reached_goal[1:]

            # 2. proceed until some agents hit their goals
            elif not self.agents_reached_goal:
                while self.path_planning_env.agents:
                    robot_state = self.path_planning_env.get_state()
                    robot_actions = self.path_planner.act(robot_state)
                    self.path_planning_env.step(robot_actions)
                    self.it += 1
                    R -= 1

                    reached_goal_n = self.path_planning_env.is_goal_state(robot_state)
                    self.agents_reached_goal = np.where(reached_goal_n)[0].tolist()
                    if self.agents_reached_goal:
                        break

                    # terminate the episode: whether all agents become idle
                    if self.verbose:
                        print(f"Currently assigned tasks: {min(self.curr_t, len(self.tasks))}")
                    final_tasks = np.array(list(self.path_planning_env.world.next_goals.values()))
                    all_tasks = np.array(list(map(len, self.path_planning_env.world.goals)))
                    if self.curr_t >= len(self.tasks) and np.all(final_tasks == all_tasks - 1):
                        # R_task += 10000
                        R += 360
                        term = True
                        break

                    # truncate the episode: exceeding max_step
                    if self.it >= 800:
                        trunc = True
                        if self.verbose:
                            print(f"Exceeding 800 steps,"
                                  f"{min(self.curr_t, len(self.tasks))} tasks were assigned,",
                                  f"finally {self.finished} robots finished")
                        break

            # 3. still some non-trivial tasks remain
            else:
                break

        if self.curr_t < len(self.tasks):
            task = self.tasks[self.curr_t]
            candidates = self.lookup_dict[self.tasks[self.curr_t]]
        else:
            task = None
            candidates = []
        obs = (task,
               candidates,
               self.path_planning_env.get_state())
        info = {}
        # r = R_task * gamma ** (self.it - T) - R_base / (1 - gamma) * (1 - gamma ** (self.it - T))
        # r = R_task * gamma ** self.it
        r = R
        # r = R_task - R_base * (self.it - T)

        if term or trunc:
            if self.verbose:
                print(f"In total, {min(self.curr_t, len(self.tasks))} tasks were assigned,",
                      f"took {len(self.path_planning_env.world.history['paths']) - 1} steps")

        return obs, r, term, trunc, info

    def render(self):
        """
        Render the history
        """
        return self.path_planning_env.world._render()

    def save(self, file_name, speed=1):
        """
        Save the visualized result

        .. note::

            Run ``conda install conda-forge::ffmpeg`` first, if ``ffmpeg`` is not installed

        Args:
            file_name (str): output file path
            speed (int): speedup rate
        """
        self.path_planning_env.world._save(file_name, speed)


class MATARP4RL(Env):
    metadata = {
        'name': 'multiagent_task_assignment_and_route_planning_for_RL_v0'
    }

    def __init__(self, matarp_env):
        super().__init__()
        self.env = matarp_env

    def reset(self, seed=None, options=None):
        """super
        obs = (self.tasks[self.curr_t],
               self.lookup_dict[self.tasks[self.curr_t]],
               self.path_planning_env.get_state())
        info = {
            'tasks': deepcopy(self.tasks),
            'lookup_dict': deepcopy(self.lookup_dict)
        }
        """
        seed = np.random.randint(0, 10)
        obs, info = self.env.reset(seed, options)
        self.layout = self.env.path_planning_env.world.layout
        candidates = obs[1]
        locations = obs[2]['locations']
        directions = obs[2]['directions']
        new_obs = np.concatenate(
            [locations / np.array(self.layout.shape),
             (np.array(directions) / 360).reshape(-1, 1)],
            axis=1
        ).reshape(-1)

        action_mask = np.zeros(151, dtype=np.int8)
        action_mask[candidates] = True
        new_info = {'action_mask': action_mask}
        self.action_mask = action_mask

        if getattr(self, 'action_space', None) is None or\
                getattr(self, 'observation_space', None) is None:
            self.action_space = spaces.Discrete(151)
            total_len_obs = len(new_obs)
            self.observation_space = spaces.Box(
                low=np.zeros(total_len_obs, dtype=np.float32),
                high=np.ones(total_len_obs, dtype=np.float32),
                dtype=np.float32
            )

        return new_obs, new_info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)

        candidates = obs[1]
        locations = obs[2]['locations']
        directions = obs[2]['directions']
        new_obs = np.concatenate(
            [locations / np.array(self.layout.shape),
             (np.array(directions) / 360).reshape(-1, 1)],
            axis=1
        ).reshape(-1)

        action_mask = np.zeros(151, dtype=np.int8)
        action_mask[candidates] = True
        new_info = {'action_mask': action_mask}
        self.action_mask = action_mask

        return new_obs, r, term, trunc, new_info

    def get_action_mask(self):
        return self.action_mask


def mask_fn(env):
    return env.get_action_mask()


if __name__ == '__main__':
    import argparse
    import time

    import supersuit as ss
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    from agents.centralized import LiBiaoRobot
    from marp.ma_env import MARP
    from marp.utils import show_args

    def get_args():
        parser = argparse.ArgumentParser(
            description='multiagent_task_assignment_and_route_planning_for_RL.'
        )
        parser.add_argument('--task-seed', dest='task_seed', type=int, default=0,
                            help='Specify a seed for task radomization')
        parser.add_argument('--robot-seed', dest='robot_seed', type=int, default=0,
                            help='Specify a seed for robot radomization')
        parser.add_argument('--it', dest='it', type=str, default='3e5',
                            help='Specify the number of training iterations')
        parser.add_argument('--config', dest='config', type=int, default=0,
                            help='Specify the configuration for training')

        args = parser.parse_args()
        return args

    args = get_args()
    show_args(args)

    configs = {
        0: dict(
            batch_size=64,
            policy_kwargs={
                'net_arch': dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128]),
            },
            n_steps=135 * 3,
        ),
        1: dict(
            batch_size=128,
            policy_kwargs={
                'net_arch': dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256]),
            },
            n_steps=135 * 3,
        ),
        2: dict(
            batch_size=128,
            policy_kwargs={
                'net_arch': dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512]),
            },
            n_steps=135 * 3,
        ),
        3: dict(
            batch_size=128,
            policy_kwargs={
                'net_arch': dict(pi=[1024, 1024, 1024, 1024], vf=[1024, 1024, 1024, 1024]),
            },
            n_steps=135 * 3,
        ),
    }

    log_file = 'related_tasks.xlsx'
    matarp_env = MATARP(MARP, LiBiaoRobot, log_file, args.task_seed, args.robot_seed, verbose=0)
    env = MATARP4RL(matarp_env)
    env.reset()

    # ep_r = 0
    # obs, info = env.reset()
    # while True:
    #     print(obs)
    #     action = env.action_space.sample(info['action_mask'])
    #     obs, r, term, trunc, info = env.step(action)
    #     print(r)
    #     ep_r += r
    #     if term or trunc:
    #         break
    # print(ep_r)

    masked_env = ActionMasker(env, mask_fn)
    training_env = ss.stable_baselines3_vec_env_v0(masked_env, num_envs=8)
    training_env.reset()
    model = MaskablePPO(
        "MlpPolicy",
        training_env,
        verbose=1,
        tensorboard_log="runs/ta",
        **configs[args.config]
    )
    eval_callback = MaskableEvalCallback(masked_env,
                                         eval_freq=135 * 6, n_eval_episodes=1, deterministic=True,
                                         best_model_save_path="best_models/ta")
    model.learn(total_timesteps=int(eval(args.it)),
                tb_log_name=f"{time.strftime('%Y-%m-%d-%H%M%S', time.localtime())}"
                            f"_MaskablePPO_cfg{args.config}",
                callback=eval_callback)
    model.save(f"pretrained/ta/MaskablePPO_cfg{args.config}_{args.it}_360_t0r0")
