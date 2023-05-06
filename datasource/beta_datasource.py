import pandas as pd
import numpy as np

from datasource import DataSource


class BetaDataSource(DataSource):
    def __init__(self, alpha=0.01, beta=2., size=1000000, seed=3212142, noise=None):
        self.random = np.random.RandomState(seed)
        proxy_scores = self.random.beta(alpha, beta, size=size)  # create scores
        true_labels = self.random.binomial(n=1, p=proxy_scores)  # create labels

        if noise is not None:
            proxy_scores = proxy_scores + self.random.normal(scale=noise, size=len(proxy_scores))  # add noise
            proxy_scores = proxy_scores.clip(0, 1)  # between 0 1

        data = {'id': list(range(size)),
                'proxy_score': proxy_scores,
                'label': true_labels}
        df = pd.DataFrame(data)
        super().__init__(df)


def get_beta(alpha, beta, noise) -> DataSource:
    return BetaDataSource(alpha=alpha, beta=beta, noise=noise)
