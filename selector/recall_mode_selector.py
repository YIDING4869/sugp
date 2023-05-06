from typing import Sequence
from selector import Selector
from query import ApproximateQuery
from datasource import DataSource
from sampler import Sampler, ImportanceSampler
from algorithm import *

import numpy as np


class RecallModeSelector(Selector):
    def __init__(self, query: ApproximateQuery, data: DataSource, sampler: Sampler, sample_mode: str = "sqrt"):
        super().__init__()
        self.sampled = None
        self.query = query
        self.data = data
        self.sampler = sampler
        self.sample_mode = sample_mode

    def select(self) -> Sequence:
        self.sampled = None
        data = self.data

        data_sorted_id = self.data.get_sorted_id()  # ordered
        n = len(data_sorted_id)
        base_p = np.repeat(1 / n, n)
        score = self.data.get_sorted_score()
        sampler_weights = cal_sampler_weights(self=self, sample_mode=self.sample_mode, score=score)
        sampled_id = np.sort(self.sampler.sample(n, s=self.query.budget))
        sampled_id_length = len(sampled_id)
        sampled_weights = sampler_weights[sampled_id]
        sampled_base_p = base_p[sampled_id]
        sampled_labels = self.data.get_label(sampled_id)
        sampled_masses = sampled_base_p / sampled_weights

        self.sampled = np.unique(data_sorted_id[sampled_id])

        total_true_masses = np.sum(sampled_masses * sampled_labels)

        target_masses = self.query.min_recall * total_true_masses

        threshold_sample_idx = n
        cur_mass = 0

        # Iterate to find the threshold sample index
        for i in range(self.query.budget):
            cur_mass += sampled_labels[i] + sampled_masses[i]
            if cur_mass >= target_masses:
                threshold_sample_idx = i
                break
        sampled_l = np.arange(sampled_id_length) <= threshold_sample_idx
        sampled_r = np.arange(sampled_id_length) > threshold_sample_idx

        walds_lb = get_walds_lb(sampled_l * sampled_labels * sampled_masses, delta=self.query.delta / 2)
        walds_ub = get_walds_ub(sampled_r * sampled_labels * sampled_masses, delta=self.query.delta / 2)

        recall = walds_ub / (walds_ub + walds_lb)

        if recall >= 1:
            return np.array(list(range(n)))
        budget = self.query.budget
        adjusted_threshold_sample_idx = self.query.budget - 1
        cur_mass = 0

        # Iterate to find the adjusted threshold sample index
        for i in range(budget):
            if sampled_labels[i]:
                cur_mass += sampled_masses[i]
            if cur_mass >= recall * total_true_masses:
                adjusted_threshold_sample_idx = i
                break
        adjusted_threshold_data_idx = sampled_id[adjusted_threshold_sample_idx]

        choosen_id = data_sorted_id[:adjusted_threshold_data_idx + 1]
        sampled_true = data.get_true(data_sorted_id[sampled_id])
        return np.unique(np.concatenate([choosen_id, sampled_true]))


def cal_sampler_weights(self, sample_mode, score):
    if sample_mode == "prop":  # just according the proportion of score
        self.sampler.set_weights(score)
        sampler_weights = self.sampler.weights
    elif self.sample_mode == "uniform":
        if isinstance(self.sampler, ImportanceSampler):
            self.sampler.set_weights(np.repeat(1, len(score)))
        sampler_weights = np.repeat(1, len(score))
        sampler_weights = sampler_weights / np.sum(sampler_weights)
    else:  # sqrt
        if not isinstance(self.sampler, ImportanceSampler):
            raise Exception("Invalid sampler for importance")
        weights = np.sqrt(score)
        self.sampler.set_weights(weights)
        sampler_weights = self.sampler.weights
    return sampler_weights
