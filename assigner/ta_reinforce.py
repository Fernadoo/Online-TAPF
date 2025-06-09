import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


# Hyperparameters
learning_rate = 0.0002
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.data = []

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, action_mask):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        print(x)
        x = F.softmax(self.fc4(x), dim=0)
        print(x)
        return x * action_mask

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        if len(self.data) < 135:
            print(self.data)
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()
        self.data = []

        return loss.item()


if __name__ == '__main__':
    import argparse
    import time

    import numpy as np

    from agents.centralized import LiBiaoRobot
    from marp.ma_env import MARP
    from marp.ta_env import MATARP, MATARP4RL
    from marp.utils import show_args

    def get_args():
        parser = argparse.ArgumentParser(
            description='multiagent_task_assignment_and_route_planning_for_RL.'
        )
        parser.add_argument('--task-seed', dest='task_seed', type=int, default=0,
                            help='Specify a seed for task radomization')
        parser.add_argument('--robot-seed', dest='robot_seed', type=int, default=0,
                            help='Specify a seed for robot radomization')
        parser.add_argument('--ep', dest='ep', type=str, default='1000',
                            help='Specify the number of training episodes')
        parser.add_argument('--config', dest='config', type=int, default=0,
                            help='Specify the configuration for training')

        args = parser.parse_args()
        return args

    args = get_args()
    show_args(args)

    configs = {
        0: dict(
            batch_size=64,
            hidden_dim=256,
            n_steps=135 * 3,
        )
    }

    log_file = 'related_tasks.xlsx'
    matarp_env = MATARP(MARP, LiBiaoRobot, log_file, args.task_seed, args.robot_seed, verbose=0)
    env = MATARP4RL(matarp_env)
    obs, info = env.reset()

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

    obs_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    logger = SummaryWriter(
        f"runs/ta/"
        f"{time.strftime('%Y-%m-%d-%H%M%S', time.localtime())}_MaskableREINFORCE"
    )

    pi = Policy(in_dim=obs_dim, hidden_dim=configs[args.config]['hidden_dim'], out_dim=a_dim)
    print(pi)

    eval_interval = 3 * 8
    best_eval_score = -999
    for n_epi in range(int(eval(args.ep))):
        eval_score = 0.
        obs, info = env.reset()
        term = False
        trunc = False

        # i = 0
        while not (term or trunc):
            if not np.any(info['action_mask']):
                break
            probs = pi(torch.from_numpy(obs).float(),
                       torch.from_numpy(info['action_mask']).float())
            print(probs)
            norm_probs = probs / probs.sum()
            if torch.all(torch.isnan(norm_probs)):
                print(obs, info)
                break
            m = Categorical(probs=norm_probs)
            a = m.sample()
            obs, r, term, trunc, info = env.step(a.item())
            pi.put_data((r, probs[a]))
            obs = obs
            eval_score += r

        #     i += 1
        #     print(i, obs.shape, r, term, trunc, np.any(info['action_mask']), eval_score)
        # exit()

        loss = pi.train_net()
        logger.add_scalar('train/loss', loss, n_epi * 135)
        logger.add_scalar('train/return', eval_score, n_epi * 135)

        if n_epi % eval_interval == 0 and n_epi != 0:
            logger.add_scalar('eval/mean_reward', eval_score, n_epi * 135)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                torch.save(pi, f'best_models/ta/best_model_rfc_cfg{args.config}_ep{args.ep}_t0r0.pt')
                print(f'New best model with eval score {eval_score}')

    torch.save(pi, f'pretrained/ta/MaskableRFC_cfg{args.config}_ep{args.ep}_t0r0.pt')
