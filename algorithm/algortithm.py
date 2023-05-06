from math import log, sqrt

import numpy as np


# （Wald's confidence interval）（based on Central Limit Theorem，CLT）
def get_walds_lb(data, delta):
    mu = np.mean(data)
    std = np.std(data)
    n = len(data)
    se = std / sqrt(n)
    z_score = sqrt(2 * log(1 / (2 * delta)))
    return mu - z_score * se


def get_walds_ub(data, delta):
    mu = np.mean(data)
    std = np.std(data)
    n = len(data)
    se = std / sqrt(n)
    z_score = sqrt(2 * log(1 / (2 * delta)))
    return mu + z_score * se
