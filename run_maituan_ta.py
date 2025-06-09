from functools import partial

from agents.centralized import LiBiaoRobot, TPTSRobot, RHCRRobot
from assigner.decoupled import RandomAssigner, CloserPortFirstAssigner, FartherPortFirstAssigner
from assigner.mpc import RLAssigner
from assigner.reactive import AlphaAssigner
from marp.ma_env import MARP
from marp.ta_env import MATARP
from marp.utils import show_args


if __name__ == '__main__':
    import argparse

    def get_args():
        parser = argparse.ArgumentParser(
            description='Multi-Agent Task Assignment and Route Planning.'
        )
        parser.add_argument('--router', dest='router', type=str, default='libiao',
                            help='choose an algorithm among [libiao, tpts, ]')
        parser.add_argument('--num', dest='num', type=int, default=50,
                            help='Specify the number of agents')
        parser.add_argument('--assigner', dest='assigner', type=str,
                            help='Choose an assigner among [random, closer, farther, alpha, mpc]')
        parser.add_argument('--alpha', dest='alpha', type=float,
                            help='Choose a threshold for the adaptive assigner')
        parser.add_argument('--model', dest='model', type=int,
                            help='Choose a pretrained model among [0, 1, 2, 3]')
        parser.add_argument('--task-seed', dest='task_seed', type=int, default=0,
                            help='Specify a seed for task radomization')
        parser.add_argument('--robot-seed', dest='robot_seed', type=int, default=0,
                            help='Specify a seed for robot radomization')
        parser.add_argument('--vis', dest='vis', action='store_true',
                            help='Visulize the process')
        parser.add_argument('--save', dest='save', type=str,
                            help='Specify the path to save the animation')

        args = parser.parse_args()
        # if args.save:
        #     args.save = 'figs/' + args.save

        return args

    args = get_args()
    show_args(args)

    if args.router == 'libiao':
        router = LiBiaoRobot
    elif args.router == 'tpts':
        router = partial(TPTSRobot, heu='m')
    elif args.router == 'tpts_r':
        router = partial(TPTSRobot, heu='r')
    elif args.router == 'rhcr':
        router = partial(RHCRRobot, heu='m')
    elif args.router == 'rhcr_r':
        router = partial(RHCRRobot, heu='r')

    log_file = 'related_tasks.xlsx'
    task_env = MATARP(MARP, router, args.num,
                      log_file, args.task_seed, args.robot_seed, verbose=1)

    obs, info = task_env.reset()
    tasks = info['tasks']  # but invisible to the assigner
    lookup_dict = info['lookup_dict']

    if args.assigner == 'random':
        assigner = RandomAssigner(lookup_dict, seed=args.task_seed)
    elif args.assigner == 'closer':
        assigner = CloserPortFirstAssigner(lookup_dict)
    elif args.assigner == 'farther':
        assigner = FartherPortFirstAssigner(lookup_dict)
    elif args.assigner == 'alpha':
        assigner = AlphaAssigner(lookup_dict, alpha=args.alpha)
    elif args.assigner == 'mpc':
        assigner = RLAssigner(lookup_dict, info=info, model=args.model)
    else:
        raise ValueError('No such assigner!')

    ep_r = 0
    while True:
        new_task, candidate_ports, state = obs
        action = assigner.assign(new_task, state=state)
        obs, r, term, trunc, info = task_env.step(action)
        # print(r)
        ep_r += r
        if term or trunc:
            break

    if args.vis:
        task_env.render()

    if args.save:
        task_env.save(args.save)
