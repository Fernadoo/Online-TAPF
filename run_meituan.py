import pickle
import time

from marp.ma_env import MARP
from agents.centralized import LiBiaoRobot, CBSRobot, TPTSRobot, PIBTRobot, RHCRRobot


def make_env(N=3, layout='small',
             orthogonal_actions=True, one_shot=True, battery=False, render_mode=None,
             **kwargs):
    return MARP(N, layout, orthogonal_actions, one_shot, battery, render_mode, **kwargs)


def append_new_goals(warehouse, new_goals):
    for agent in new_goals:
        i = eval(agent.split('_')[-1])
        warehouse.world.goals[i] = warehouse.world.goals[i] + new_goals[agent]


if __name__ == '__main__':
    import argparse
    import numpy as np
    import pandas as pd
    import random
    from itertools import product

    def get_args():
        parser = argparse.ArgumentParser(
            description='Multi-Agent Route Planning.'
        )
        parser.add_argument('--rotate', dest='rotate', action='store_true',
                            help='Visulize the process')
        parser.add_argument('--load-tasks', dest='load_tasks', type=str,
                            help='Load a sequence of tasks from a pickled file')
        parser.add_argument('--sorted', dest='sorted', action='store_true',
                            help='Sort the given task file by drop-time')
        parser.add_argument('--random', dest='random', type=int, default=0,
                            help='Randomize goals')
        parser.add_argument('--num', dest='num', type=int, default=50,
                            help='specify the number of agents')
        parser.add_argument('--alg', dest='alg', type=str, default='libiao',
                            help='choose an algorithm among [libiao, cbs, ]')
        parser.add_argument('--seed', dest='seed', type=int, default=0,
                            help='Specify a seed for PNGs')
        parser.add_argument('--vis', dest='vis', action='store_true',
                            help='Visulize the process')
        parser.add_argument('--save', dest='save', type=str,
                            help='Specify the path to save the animation')

        args = parser.parse_args()
        if args.save:
            args.save = 'figs/' + args.save

        return args

    args = get_args()
    random.seed(args.seed)

    possible_starts = list(product(range(1, 5), range(1, 82)))
    starts = random.sample(possible_starts, k=args.num)

    possible_goals = list(product([1], range(8, 82))) + list(product([4], range(81, 5, -1)))

    if args.load_tasks:
        with open(args.load_tasks, 'rb') as f:
            tasks = pickle.load(f)
        tasks = [possible_goals[t - 1] for t in tasks]
    else:
        if args.random:
            tasks = random.sample(possible_goals, k=args.random)
        else:
            # records = pd.read_excel(
            #     'records.xlsx',
            #     sheet_name='pyexcel_sheet1'
            # )
            records = pd.read_excel(
                'related_tasks.xlsx',
                sheet_name='落格记录'
            )
            if args.sorted:
                records = records.sort_values(by="落格时间", ascending=True)
            tasks = list(records['格口'])
            tasks = [possible_goals[t - 1] for t in tasks]
    curr_t = 0
    print(len(tasks), tasks[:5])

    """
    random check
    """
    # env = make_env(N=3, layout='meituan_warehouse',
    #                orthogonal_actions=False, one_shot=False, battery=True, render_mode='human',
    #                **kwargs)
    # observations, infos = env.reset()
    # for _ in range(10):
    #     actions = {
    #         agent: env.action_space(agent).sample(np.array(infos[agent]['action_mask'], dtype=np.int8))
    #         for agent in env.agents
    #     }
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(len(env.world.history['paths']), len(env.world.history['batteries']))
    # env.render()

    """
    single agent search
    """
    # env = make_env(N=3, layout='meituan_warehouse',
    #                orthogonal_actions=False, one_shot=False, battery=True, render_mode='human',
    #                **kwargs)
    # env.reset()
    # from marp.search import astar, SingleAgentRechargeLifelongSearchWrapper
    # plans = {}
    # for agent in env.agents:
    #     plans[agent] = astar(SingleAgentRechargeLifelongSearchWrapper(env, agent))
    #     print(plans[agent])
    # while env.agents:
    #     actions = {}
    #     for agent in env.agents:
    #         path = plans[agent]
    #         if len(path):
    #             actions[agent] = path.pop(0)
    #         else:
    #             actions[agent] = 0
    #         # actions['robot_0'] = 3
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    # print(env.world.history['batteries'])
    # env.render()

    if args.rotate:
        """
        50-agent case: rorating robots
        """
        # env = make_env(N=50, layout='meituan_warehouse',
        #                orthogonal_actions=False, one_shot=False, battery=True, render_mode='human',
        #                **kwargs)
        # print(goals)
        # observations, infos = env.reset()
        # for _ in range(18):
        #     actions = {
        #         agent: env.action_space(agent).sample(
        #             np.array(infos[agent]['action_mask'], dtype=np.int8)
        #         )
        #         for agent in env.agents
        #     }
        #     print(actions)
        #     observations, rewards, terminations, truncations, infos = env.step(actions)
        # env.render()
        # # env.save('meituan_warehouse_R.mp4')

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
        env = make_env(N=args.num, layout='warehouse150_1',
                       orthogonal_actions=False, one_shot=False, battery=True, render_mode='human',
                       **kwargs)

        observations, infos = env.reset()
        if args.alg == 'libiao':
            planner = LiBiaoRobot(env)
        elif args.alg == 'cbs':
            planner = CBSRobot(env)
        elif args.alg == 'tpts':
            planner = TPTSRobot(env, heu='m')
        elif args.alg == 'tpts_r':
            planner = TPTSRobot(env, heu='r')
        elif args.alg == 'pibt':
            planner = PIBTRobot(env)
        elif args.alg == 'rhcr':
            planner = RHCRRobot(env, heu='m')
        elif args.alg == 'rhcr_r':
            planner = RHCRRobot(env, heu='r')

        t0 = time.time()
        it = 0
        delivered = 0
        finished = 0
        while env.agents:
            state = env.get_state()
            actions = planner.act(state)
            # print(actions)

            # random no-op execution: due to 1) unsuccessful execution 2) unloading cargos
            # for agent in actions:
            #     if random.random() < 0.1:
            #         actions[agent] = 0

            observations, rewards, terminations, truncations, infos = env.step(actions)

            # will first stop for a unit-step at the previous pickup
            reached_goal_n = env.is_goal_state(state)
            agents_reached_goal = np.where(reached_goal_n)[0]
            # print(agents_reached_goal)
            next_k = len(agents_reached_goal)

            new_goals = tasks[curr_t: curr_t + next_k]
            curr_t += next_k

            for i, ag_i in enumerate(agents_reached_goal):
                if i >= len(new_goals):  # for the last round
                    new_task = [(-1, -1)]  # for dummy destination
                    finished += 1
                elif new_goals[i][0] == 1:
                    pickup = (1, 83)
                    new_task = [new_goals[i], pickup]
                    delivered += 1
                elif new_goals[i][0] == 4:
                    pickup = (1, 84)
                    new_task = [new_goals[i], pickup]
                    delivered += 1
                env.world._append_new_goals({f'robot_{ag_i}': new_task})

            print(f"Currently assigned tasks: {min(curr_t, len(tasks))}")
            final_tasks = np.array(list(env.world.next_goals.values()))
            all_tasks = np.array(list(map(len, env.world.goals)))

            # print(np.where(final_tasks - all_tasks + 1))
            # if len(np.where(final_tasks - all_tasks + 1)[0]) > 0:
            #     print(final_tasks[np.where(final_tasks - all_tasks + 1)])
            #     for j in range(len(np.where(final_tasks - all_tasks + 1)[0])):
            #         print(env.world.goals[np.where(final_tasks - all_tasks + 1)[0][j]])

            if curr_t >= len(tasks) and np.all(final_tasks == all_tasks - 1):
                break

            it += 1
            if it >= 800:
                print(f"Exceeding 800 steps, {min(curr_t, len(tasks))} tasks were assigned,",
                      # f"{delivered} tasks were delivered,",
                      f"finally {finished} robots finished")
                break

        if getattr(planner, 'alg_hist', None):
            print(planner.alg_hist)
        print(f"In total, {min(curr_t, len(tasks))} tasks were assigned,",
              # f"{delivered} tasks were delivered,",
              f"took {len(env.world.history['paths']) - 1} steps")

        t1 = time.time()
        print(f"took {t1 - t0}")

        if args.vis:
            env.render()
        # env.save('meituan_warehouse_R_rules.mp4')

    else:
        """
        50-agent case: standard robots
        """
        from marp.animator import WarehouseAnimation
        rand_goals = random.sample(possible_goals, k=50)
        goals = [[tuple(g)] for g in rand_goals]
        kwargs = {
            'starts': starts,
            'goals': goals,
            'full_battery': np.inf
        }
        print(starts)
        print(goals)
        env = make_env(N=50, layout='warehouse150_1',
                       orthogonal_actions=True, one_shot=False, battery=True, render_mode='human',
                       **kwargs)
        observations, infos = env.reset()
        cbsr = CBSRobot(env)
        state = env.get_state()
        paths = cbsr.act(state)

        # align paths
        maxlen = max(list(map(len, paths)))
        history = []
        for i in range(maxlen):
            state = []
            for ai in range(50):
                if i >= len(paths[ai]):
                    state.append(paths[ai][-1])
                else:
                    state.append(paths[ai][i])
            history.append(state)
        dummy_bat = []
        for i in range(maxlen):
            dummy_bat.append([np.inf for _ in range(50)])
        animation = WarehouseAnimation(
            range(50),
            env.world.layout,
            env.world.starts,
            env.world.goals,
            history,
            dummy_bat,
            FPS=15)
        if args.save:
            animation.show()
        # animation.save('figs/meituan_warehouse_cbs.mp4', speed=2, dpi=200)

    # for _ in range(1):
    #     actions = {
    #         agent: env.action_space(agent).sample(np.array(infos[agent]['action_mask'], dtype=np.int8))
    #         for agent in env.agents
    #     }
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    # env.render()
    # env.save('meituan_warehouse.mp4')
