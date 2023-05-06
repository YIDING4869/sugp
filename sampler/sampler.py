import numpy as np


class Sampler:
    def __init__(self, seed=0):
        self.random = np.random.RandomState(seed)

    def reset(self):
        pass

    def sample(self, max_id: int, s: int):
        if s > max_id:
            return self.random.choice(max_id, size=max_id, replace=False)
        else:
            return self.random.choice(max_id, size=s, replace=False)

    def set_weights(self, weights):
        pass
