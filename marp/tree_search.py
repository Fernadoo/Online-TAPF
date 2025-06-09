from copy import deepcopy


class TreeSearchWrapper():
    """
    A tree search wrapper should provide both search and rl APIs
    """

    def __init__(self, ma_env):
        ma_env.reset()
        self.ma_env = ma_env
        self.world = ma_env.world
        self.agents = ma_env.agents
        self.REWARDS = deepcopy(self.ma_env.world.REWARDS)

    def get_state(self):
        return self.ma_env.get_state()

    def transit(self, state, actions):
        if isinstance(actions, tuple) or isinstance(actions, list):
            actions = dict(zip(self.agents, actions))
        succ_state, collision_free = self.ma_env.transit(state, actions)
        rewards = {}
        is_goal_n = self.is_goal_state(succ_state)
        if collision_free:
            for i, agent in enumerate(self.agents):
                if is_goal_n[i]:
                    rewards[agent] = self.REWARDS['goal']
                else:
                    rewards[agent] = self.REWARDS['normal']
            return succ_state, rewards
        else:
            for i, agent in enumerate(self.agents):
                if succ_state['infos'][agent]['collide_with']:
                    rewards[agent] = self.REWARDS['collision']
                else:
                    rewards[agent] = self.REWARDS['normal']
            return succ_state, rewards

    def is_goal_state(self, state):
        return self.ma_env.is_goal_state(state)

    def reset(self, seed=None, options=None):
        observations, infos = self.ma_env.reset(seed=None, options=None)
        self.agents = self.ma_env.agents
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.ma_env.step(actions)
        self.agents = self.ma_env.agents
        return observations, rewards, terminations, truncations, infos

    def action_space(self, agent):
        return self.ma_env.action_space(agent)
