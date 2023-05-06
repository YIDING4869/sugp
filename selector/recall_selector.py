from typing import Sequence
from selector import Selector
from query import ApproximateQuery
from datasource import DataSource
from sampler import Sampler, ImportanceSampler

import numpy as np


class RecallSelector(Selector):
    def __init__(self, query: ApproximateQuery, data: DataSource, sampler: Sampler):
        super().__init__()
        self.query = query
        self.data = data
        self.sampler = sampler

    def select(self) -> Sequence:
        data_sorted_id = self.data.get_sorted_id()  # ordered
        n = len(data_sorted_id)

        if isinstance(self.sampler, ImportanceSampler):
            self.sampler.set_weights(np.repeat(1, n))

        sampled_id = self.sampler.sample(max_id=n, s=self.query.budget)
        id = data_sorted_id[sampled_id]

        label = self.data.get_label(id)
        score = self.data.get_score(id)
        sorted_tuples = sorted(list(zip(score, label, list(range(self.query.budget)))), reverse=True)
        sorted_list = list(sorted_tuples)

        sampled_true = self.data.get_true(sampled_id)
        sampled_true_number = len(sampled_true)
        satisfied_id = satisfy_recall(sorted_list=sorted_list, query=self.query, sampled_id=sampled_id,
                                      sample_true_number=sampled_true_number)

        choosen_id = data_sorted_id[:satisfied_id]
        all_true = self.data.get_true(id)
        return np.unique(np.concatenate([choosen_id, all_true]))


def satisfy_recall(sorted_list, query, sampled_id, sample_true_number):
    choosen = [-1]
    true_number = 0.
    for n in range(query.budget):
        true_number += sorted_list[n][1]
        recall = true_number / sample_true_number
        if recall > query.min_recall:
            choosen.append(n)
            break
    satisfied_id = sampled_id[sorted_list[choosen[-1]][2]]
    return satisfied_id
