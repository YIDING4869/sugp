from sampler import Sampler

import numpy as np


class ImportanceSampler(Sampler):
    def __init__(self, seed=0, uniform_weight=0.10):
        super().__init__(seed)
        self.uniform_weight = uniform_weight
        self.seed = 0
        self.random = np.random.RandomState(seed)
        self.weights = None
        self.sampled_idxs = None
        self.sampled_weights = None

    def reset(self):
        self.weights = None
        self.sampled_idxs = None
        self.sampled_weights = None

    def set_weights(self, weights: np.ndarray):
        scaled_probability = weights / np.sum(weights)
        uniform_probability = 1 / len(weights)
        self.weights = scaled_probability * (1 - self.uniform_weight) + uniform_probability * self.uniform_weight

    def sample(self, max_id: int, s: int):
        if s > max_id:
            s = max_id
        if max_id != len(self.weights):
            weights = self.weights[:max_id]
            weights /= weights.sum()
        else:
            weights = self.weights
        self.sampled_idxs = self.random.choice(max_id, size=s, replace=True, p=weights)
        self.sampled_weights = self.weights[self.sampled_idxs]
        return self.sampled_idxs

    def get_sample_weights(self):
        return self.sampled_weights
