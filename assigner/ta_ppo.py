if __name__ == '__main__':
    import argparse
    import os
    import time

    import supersuit as ss
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    from agents.centralized import LiBiaoRobot
    from marp.ma_env import MARP
    from marp.ta_env import MATARP, MATARP4RL, mask_fn
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
        parser.add_argument('--num', dest='num', type=int, default=50,
                            help='Specify the number of agents')
        parser.add_argument('--train-rand', dest='train_rand', action='store_true',
                            help='Whether to train over randomized sequences')
        parser.add_argument('--eval-rand', dest='eval_rand', action='store_true',
                            help='Whether to evaluate over randomized sequences')

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
    matarp_env = MATARP(MARP, LiBiaoRobot, args.num, log_file, args.task_seed, args.robot_seed,
                        verbose=0, random=args.train_rand)
    matarp_env_eval = MATARP(MARP, LiBiaoRobot, args.num, log_file, args.task_seed, args.robot_seed,
                             verbose=0, random=args.eval_rand)
    env = MATARP4RL(matarp_env)
    env.reset()
    env_eval = MATARP4RL(matarp_env_eval)
    env_eval.reset()

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
    training_env = ss.stable_baselines3_vec_env_v0(masked_env, num_envs=8, multiprocessing=False)
    training_env.reset()
    model = MaskablePPO(
        "MlpPolicy",
        training_env,
        verbose=1,
        tensorboard_log=f"runs/ta/{args.num}a",
        device='auto',
        **configs[args.config]
    )

    # periodic evaluation
    best_model_path = (f"best_models/ta/{args.num}a/cfg{args.config}"
                       f"_t{args.task_seed}r{args.robot_seed}"
                       f"_train_{'rand' * args.train_rand + 'fix' * (not args.train_rand)}"
                       f"_eval_{'rand' * args.eval_rand + 'fix' * (not args.eval_rand)}"
                       f"/{time.strftime('%Y-%m-%d-%H%M%S', time.localtime())}")
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    masked_env_eval = ActionMasker(env_eval, mask_fn)
    evaluation_env = ss.stable_baselines3_vec_env_v0(masked_env_eval, num_envs=1, multiprocessing=False)
    evaluation_env.reset()
    eval_callback = MaskableEvalCallback(evaluation_env,
                                         eval_freq=135 * 6, n_eval_episodes=1, deterministic=True,
                                         best_model_save_path=best_model_path)

    # tb logging
    tb_log_name = (f"{time.strftime('%Y-%m-%d-%H%M%S', time.localtime())}"
                   f"_MaskablePPO_cfg{args.config}"
                   f"_t{args.task_seed}r{args.robot_seed}"
                   f"_train_{'rand' * args.train_rand + 'fix' * (not args.train_rand)}"
                   f"_eval_{'rand' * args.eval_rand + 'fix' * (not args.eval_rand)}")
    model.learn(total_timesteps=int(eval(args.it)),
                tb_log_name=tb_log_name,
                callback=eval_callback)

    # final model save
    model_path = (f"pretrained/ta/{args.num}a/MaskablePPO_cfg{args.config}_iter{args.it}"
                  f"_t{args.task_seed}r{args.robot_seed}"
                  f"_train_{'rand' * args.train_rand + 'fix' * (not args.train_rand)}"
                  f"_eval_{'rand' * args.eval_rand + 'fix' * (not args.eval_rand)}")
    model.save(model_path)
