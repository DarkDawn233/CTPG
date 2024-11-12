import gym
import numpy as np
import random

class SeedWrapper(gym.Wrapper):
    
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.env.seed(seed)