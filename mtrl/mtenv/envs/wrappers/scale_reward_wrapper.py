import gym

class ScaleRewardWrapper(gym.Wrapper):

    def __init__(self, env, reward_scale=1):
        super(ScaleRewardWrapper, self).__init__(env)
        self._reward_scale = reward_scale

    def reward(self, reward):
        return self._reward_scale * reward
        
    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        return obs, self.reward(rew), done, info