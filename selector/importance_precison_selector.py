from typing import Sequence
from selector import Selector
from query import ApproximateQuery
from datasource import DataSource
from sampler import Sampler, ImportanceSampler
from algorithm import *

import numpy as np


class ImportancePrecisionSelector(Selector):
    def __init__(self, query: ApproximateQuery, data: DataSource, sampler: Sampler, start=100, step=100):
        super().__init__()
        self.query = query
        self.data = data
        self.sampler = sampler
        self.start = start
        self.step = step

    def select(self) -> Sequence:
        data_sorted_id = self.data.get_sorted_id()  # ordered
        n = len(data_sorted_id)
        T = 1 + (self.query.budget - self.start) // self.step

        if isinstance(self.sampler, ImportanceSampler):
            self.sampler.set_weights(np.repeat(1, n))

        sampled_id = np.sort(self.sampler.sample(max_id=n, s=self.query.budget))
        id = data_sorted_id[sampled_id]
        label = self.data.get_label(id)
        delta = self.query.delta

        sampler_weights = self.sampler.weights
        base_p = np.repeat(1.0 / n, n)
        # satisfied_id=satisfy_precison(query=self.query,sampled_id=sampled_id,label=label,start=self.start,step=self.step,delta=delta/T,base_p=base_p,sampler_weights=sampler_weights)
        choosen = [0]
        for n in range(self.start, self.query.budget, self.step):
            cur = sampled_id[n]
            cur_base_p = base_p[:cur + 1] / np.sum(base_p[:cur + 1])
            cur_sampler_weights = sampler_weights[:cur + 1] / np.sum(sampler_weights[:cur + 1])
            cur_masses = cur_base_p / cur_sampler_weights
            # Get the current subsample of sampled indices
            cur_subsample = sampled_id[:n + 1]
            data = label[:n + 1] * cur_masses[cur_subsample]

            walds_lb = get_walds_lb(data, delta/T)
            if walds_lb > self.query.min_precision:
                choosen.append(n)

        choosen_id = data_sorted_id[:choosen[-1]]
        all_true = self.data.get_true(id)
        return np.unique(np.concatenate([choosen_id, all_true]))
