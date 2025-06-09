from marp.ma_env import MARP


def make_env(N=3, layout='small',
             orthogonal_actions=True, one_shot=True, battery=False, render_mode=None,
             **kwargs):
    return MARP(N, layout, orthogonal_actions, one_shot, battery, render_mode, **kwargs)


if __name__ == '__main__':
    #######
    # MAPF TESTING
    #######

    env = make_env(N=3, render_mode='human')

    """
    Env testing
    """
    # observations, infos = env.reset()
    # print(observations)
    # print(infos)
    # print()
    # for _ in range(30):
    #     actions = {
    #         agent: env.action_space(agent).sample(infos[agent]['action_mask'])
    #         for agent in env.agents
    #     }
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(observations)
    #     print(actions)
    #     print(rewards)
    #     print(terminations)
    #     print(truncations)
    #     print(infos)
    #     print()
    # env.render()

    """
    Single agent search testing
    """
    # from copy import deepcopy
    # env.reset()
    # from marp.search import SingleAgentSearchWrapper, astar
    # plans = {}
    # for agent in env.agents:
    #     plans[agent] = astar(SingleAgentSearchWrapper(env, agent))
    # print(plans)
    # _plans = deepcopy(plans)
    # while env.agents:
    #     actions = {}
    #     for agent in env.agents:
    #         path = plans[agent]
    #         if len(path):
    #             actions[agent] = path.pop(0)
    #         else:
    #             actions[agent] = 0
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(observations)
    #     print(actions)
    #     print(rewards)
    #     print(terminations)
    #     print(truncations)
    #     print(infos)
    # env.render()
    # env.save('sapf.mp4', speed=1)

    """
    Multi agent joint search testing
    """
    # env.reset()
    # from marp.search import MultiAgentSearchWrapper, astar
    # joint_plan = astar(MultiAgentSearchWrapper(env))
    # print(joint_plan)
    # while env.agents:
    #     actions = joint_plan.pop(0)
    #     env.step(actions)
    # env.render()
    # env.save('mapf.mp4')

    """
    RL testing ma-il
    """
    # env.reset()
    # import time
    # import supersuit as ss
    # from stable_baselines3 import PPO, A2C, DQN

    # # model training
    # alg = DQN
    # training_env = ss.pettingzoo_env_to_vec_env_v1(env)
    # training_env = ss.concat_vec_envs_v1(training_env, 8, num_cpus=1, base_class="stable_baselines3")
    # training_env.reset()
    # model = alg("MlpPolicy", training_env,
    #             verbose=1,
    #             tau=0.8,
    #             exploration_fraction=0.2,
    #             batch_size=256,
    #             tensorboard_log="runs")
    # model.learn(total_timesteps=int(8e6),
    #             tb_log_name=f"{time.strftime('%Y-%m-%d-%H%M%S', time.localtime())}")
    # model.save("pretrained/DQN_c1e1_g1e4")

    # # model testing
    # model = alg.load("pretrained/DQN_c1e1_g1e4.zip")
    # observations, infos = env.reset()
    # while env.agents:
    #     actions = {
    #         agent: model.predict(observations[agent], deterministic=True)[0]
    #         for agent in env.agents
    #     }
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(actions)
    #     print(observations)
    #     print(rewards)
    # env.render()

    """
    RL testing ma-jl
    """
    # env.reset()
    # import time
    # import numpy as np
    # import supersuit as ss
    # from stable_baselines3 import PPO, A2C, DQN

    # from marp.rl import MultiAgentJointLearningWrapper

    # env.reset()
    # # alg = PPO
    # # policy_kwargs = {
    # #     'net_arch': dict(pi=[16, 6], vf=[16, 6]),
    # # }
    # alg = DQN
    # policy_kwargs = {
    #     'net_arch': [32, 6],
    # }
    # training_env = MultiAgentJointLearningWrapper(env)
    # ja = training_env.joint_actions
    # training_env = ss.stable_baselines3_vec_env_v0(training_env, num_envs=8)
    # training_env.reset()
    # model = alg("MlpPolicy", training_env,
    #             verbose=1,
    #             gamma=0.9,
    #             tau=0.8,
    #             batch_size=128,
    #             exploration_fraction=0.25,
    #             policy_kwargs=policy_kwargs,
    #             tensorboard_log="runs")
    # model.learn(total_timesteps=int(10e6),
    #             tb_log_name=f"{time.strftime('%Y-%m-%d-%H%M%S', time.localtime())}")
    # model.save("pretrained/jointDQN")

    # # model testing
    # model = alg.load("pretrained/jointDQN.zip")
    # observations, infos = env.reset()
    # while env.agents:
    #     a, _ = model.predict(
    #         observations[env.agents[0]]
    #     )
    #     print(a)
    #     actions = ja[a]
    #     actions = dict(zip(env.agents, actions))
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(actions)
    #     print(observations)
    #     print(rewards)
    # env.render()

    """
    RL testing sa-rl
    """
    # env.reset()
    # import time
    # import numpy as np
    # import supersuit as ss
    # from stable_baselines3 import DQN

    # from marp.rl import SingleAgentLearningWrapper

    # env.reset()
    # alg = DQN
    # policy_kwargs = {
    #     'net_arch': [8, 2],
    # }
    # for agent in env.agents:
    #     if agent != 'robot_0':
    #         continue
    #     training_env = SingleAgentLearningWrapper(env, agent)
    #     training_env = ss.stable_baselines3_vec_env_v0(training_env, num_envs=8)
    #     training_env.reset()
    #     model = alg("MlpPolicy", training_env,
    #                 verbose=1,
    #                 tau=0.5,
    #                 exploration_fraction=0.5,
    #                 batch_size=256,
    #                 policy_kwargs=policy_kwargs,
    #                 tensorboard_log="runs")
    #     model.learn(total_timesteps=int(2.5e6),
    #                 tb_log_name=f"{time.strftime('%Y-%m-%d-%H%M%S', time.localtime())}")
    #     model.save(f"pretrained/singleDQN_{agent}")

    # model testing
    # policies = {}
    # for agent in env.agents:
    #     if agent != 'robot_0':
    #         continue
    #     policies[agent] = alg.load(f"pretrained/singleDQN_{agent}.zip")
    # observations, infos = env.reset()
    # while env.agents:
    #     actions = {
    #         'robot_0': policies['robot_0'].predict(observations['robot_0'], deterministic=True)[0]
    #     }
    #     actions['robot_1'] = 0
    #     actions['robot_2'] = 0
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(actions)
    #     print(observations)
    #     print(rewards)
    # env.render()

    """
    Single agent Q learning
    """
    from marp.rl import SingleAgentLearningWrapper, Qlearning
    env.reset()
    ctrl_agent = 'robot_1'
    policy, Qtable = Qlearning(SingleAgentLearningWrapper(env, ctrl_agent))

    observations, infos = env.reset()
    while env.agents:
        a = policy[str(observations[ctrl_agent])] if str(observations[ctrl_agent]) in policy else 0
        print(Qtable[str(observations[ctrl_agent])], a)
        actions = {
            agent: a if agent == ctrl_agent else 0
            for agent in env.agents
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(actions)
        print(observations)
        print(rewards)
    env.render()
    # env.save('satql.mp4')

    """
    Multi agent joint Q learning
    """
    # from marp.rl import MultiAgentJointLearningWrapper, Qlearning
    # import pickle
    # env.reset()
    # training_env = MultiAgentJointLearningWrapper(env)
    # ja = training_env.joint_actions
    # policy = Qlearning(training_env, num_it=3e4, alpha=0.8)
    # with open("pretrained/JQL.pkl", 'wb') as f:
    #     pickle.dump(policy, f)
    # with open("pretrained/JQL.pkl", 'rb') as f:
    #     policy = pickle.load(f)
    # observations, infos = env.reset()
    # while env.agents:
    #     a = policy[str(observations[env.agents[0]])] if str(observations[env.agents[0]]) in policy else 0
    #     actions = dict(zip(env.agents, ja[a]))
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(actions)
    #     print(observations)
    #     print(rewards)
    # env.render()
    # env.save('majql.mp4')

    """
    Multi agent individual Q learning
    """
    # from marp.rl import individualQlearning
    # import pickle
    # env.reset()
    # # policies = individualQlearning(env, num_it=5e4, epsilon=0.5, alpha=0.3)
    # # with open("pretrained/IQL.pkl", 'wb') as f:
    # #     pickle.dump(policies, f)
    # with open("pretrained/IQL.pkl", 'rb') as f:
    #     policies = pickle.load(f)
    # observations, infos = env.reset()
    # while env.agents:
    #     actions = {}
    #     for agent in env.agents:
    #         if str(observations[agent]) not in policies[agent]:
    #             actions[agent] = 0
    #         else:
    #             actions[agent] = policies[agent][str(observations[agent])]
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(actions)
    #     print(observations)
    #     print(rewards)
    # env.render()
    # env.save('maiql.mp4')

    #######
    # MAPD TESTING
    #######

    # env = make_env(N=3, one_shot=False, render_mode='human')
    """
    Single-agent search
    """
    # env.reset()
    # from marp.search import SingleAgentLifelongSearchWrapper, astar
    # plans = {}
    # for agent in env.agents:
    #     plans[agent] = astar(SingleAgentLifelongSearchWrapper(env, agent))
    #     print(plans[agent])
    # while env.agents:
    #     actions = {}
    #     for agent in env.agents:
    #         path = plans[agent]
    #         if len(path):
    #             actions[agent] = path.pop(0)
    #         else:
    #             actions[agent] = 0
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(observations)
    #     print(actions)
    #     print(rewards)
    #     print(terminations)
    #     print(truncations)
    #     print(infos)
    # env.render()
    # env.save('mapd.mp4')

    """
    Multi-agent search
    """
    # env.reset()
    # from marp.search import MultiAgentLifelongSearchWrapper, astar
    # joint_plan = astar(MultiAgentLifelongSearchWrapper(env))
    # print(joint_plan)
    # while env.agents:
    #     actions = joint_plan.pop(0)
    #     env.step(actions)
    # env.render()
    # env.save('mapd_nc.mp4')

    #######
    # Warehouse TESTING
    #######

    # env = make_env(N=3, layout='small_warehouse',one_shot=False, battery=True, render_mode='human')
    """
    Single-agent search
    """
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
    # print(len(env.world.history['paths']), len(env.world.history['batteries']))
    # env.render()
    # env.save('tmp.mp4')

    """
    Multi-agent search
    """
    # env.reset()
    # from marp.search import MultiAgentRechargeLifelongSearchWrapper, astar
    # joint_plan = astar(MultiAgentRechargeLifelongSearchWrapper(env))
    # print(joint_plan)
    # # joint_plan = [{'robot_0': 3, 'robot_1': 3, 'robot_2': 4}, {'robot_0': 2, 'robot_1': 4, 'robot_2': 4}, {'robot_0': 2, 'robot_1': 3, 'robot_2': 3}, {'robot_0': 3, 'robot_1': 3, 'robot_2': 4}, {'robot_0': 2, 'robot_1': 2, 'robot_2': 3}, {'robot_0': 2, 'robot_1': 4, 'robot_2': 4}, {'robot_0': 2, 'robot_1': 1, 'robot_2': 4}, {'robot_0': 0, 'robot_1': 4, 'robot_2': 3}, {'robot_0': 2, 'robot_1': 4, 'robot_2': 3}, {'robot_0': 2, 'robot_1': 3, 'robot_2': 3}, {'robot_0': 2, 'robot_1': 3, 'robot_2': 2}, {'robot_0': 3, 'robot_1': 2, 'robot_2': 0}, {'robot_0': 0, 'robot_1': 2, 'robot_2': 0}, {'robot_0': 0, 'robot_1': 1, 'robot_2': 0}, {'robot_0': 2, 'robot_1': 2, 'robot_2': 0}, {'robot_0': 2, 'robot_1': 2, 'robot_2': 0}, {'robot_0': 2, 'robot_1': 1, 'robot_2': 0}, {'robot_0': 0, 'robot_1': 2, 'robot_2': 0}]
    # while env.agents:
    #     actions = joint_plan.pop(0)
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(env.is_goal_state(env.get_state()))
    # env.render()
    # env.save('warehouse_recharge_nc.mp4')

    #####
    # MAPF_R TESTING
    #####

    # env = make_env(N=3, layout='small', orthogonal_actions=False, render_mode='human')
    # env.reset()
    # from marp.search import SingleAgentSearchWrapper, astar
    # plans = {}
    # for agent in env.agents:
    #     plans[agent] = astar(SingleAgentSearchWrapper(env, agent))
    # print(plans)
    # while env.agents:
    #     actions = {}
    #     for agent in env.agents:
    #         path = plans[agent]
    #         if len(path):
    #             actions[agent] = path.pop(0)
    #         else:
    #             actions[agent] = 0
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(observations)
    #     print(actions)
    #     print(rewards)
    #     print(terminations)
    #     print(truncations)
    #     print(infos)
    # print(env.world.history['paths'])
    # print(env.world.history['directions'])
    # env.render()

    #####
    # MAPD_R TESTING
    #####

    # env = make_env(N=3, layout='small', orthogonal_actions=False, one_shot=False, render_mode='human')
    # env.reset()

    # from marp.search import SingleAgentLifelongSearchWrapper, astar
    # plans = {}
    # for agent in env.agents:
    #     plans[agent] = astar(SingleAgentLifelongSearchWrapper(env, agent))
    #     print(plans[agent])
    # while env.agents:
    #     actions = {}
    #     for agent in env.agents:
    #         path = plans[agent]
    #         if len(path):
    #             actions[agent] = path.pop(0)
    #         else:
    #             actions[agent] = 0
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     print(observations)
    #     print(actions)
    #     print(rewards)
    #     print(terminations)
    #     print(truncations)
    #     print(infos)
    # env.render()
    # env.save('mapd_R.mp4')

    #####
    # Warehouse_R TESTING
    #####

    # env = make_env(N=3, layout='small_warehouse', orthogonal_actions=False, one_shot=False, battery=True, render_mode='human')
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
    # print(len(env.world.history['paths']), len(env.world.history['batteries']))
    # env.render()
    # env.save('warehouse_R_recharge.mp4')
