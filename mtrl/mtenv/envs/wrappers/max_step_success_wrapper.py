import gym

class MaxStepSuccessWrapper(gym.Wrapper):

    def __init__(self, env, max_step=100):
        super(MaxStepSuccessWrapper, self).__init__(env)
        self._max_step = max_step
        self._now_step = 0
    
    def reset(self, **kwargs):
        self._now_step = 0
        return self.env.reset(**kwargs)
    
    def update_step_info(self, info):
        info.update({'env_step': self._now_step})
        if self._now_step >= self._max_step:
            info.update({'success': True})
        else:
            info.update({'success': False})
        return info

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        self._now_step += 1
        if self._now_step >= self._max_step:
            done = True
        return obs, rew, done, self.update_step_info(info)