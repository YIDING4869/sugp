from typing import Sequence

from selector.recall_mode_selector import RecallModeSelector

import numpy as np


class JointSelector(RecallModeSelector):
    def select(self) -> Sequence:
        all = super().select()
        sampled = self.sampled

        left = np.setdiff1d(all, sampled, assume_unique=True)

        self.total_sampled = len(left) + self.query.budget

        return self.data.get_true(left)
