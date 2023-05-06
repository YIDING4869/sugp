from typing import Sequence
from selector import Selector
from query import ApproximateQuery
from datasource import DataSource
from sampler import Sampler, ImportanceSampler
from algorithm import *

import numpy as np


class UniformPrecisionSelector(Selector):
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

        sampled_id = self.sampler.sample(max_id=n, s=self.query.budget)
        id = data_sorted_id[sampled_id]

        label = self.data.get_label(id)
        score = self.data.get_score(id)
        sorted_tuples = sorted(list(
            zip(score, label, list(range(self.query.budget)))), reverse=True)
        sorted_list = list(sorted_tuples)

        delta = self.query.delta

        satisfied_id = satisfy_precision(sorted_list=sorted_list, query=self.query, sampled_id=sampled_id,
                                         start=self.start, step=self.step, delta=delta / T)

        choosen_id = data_sorted_id[:satisfied_id]
        all_true = self.data.get_true(id)
        return np.unique(np.concatenate([choosen_id, all_true]))


def satisfy_precision(sorted_list, query, sampled_id, start, step, delta):
    choosen = [0]
    for n in range(start, query.budget, step):
        data = np.array([x[1] for x in sorted_list[0:n]])
        walds_lb = get_walds_lb(data, delta)
        if walds_lb > query.min_precision:
            choosen.append(n)
    if choosen[-1] == 0:
        satisfied_id = 0
    else:
        satisfied_id = sampled_id[sorted_list[choosen[-1]][2]]
    return satisfied_id
