from runner import config


class ApproximateQuery:
    def __init__(
            self,
            query_type: str = "pt",  # pt: precision threshold   rt: recall threshold  prt: both
            min_precision=None,
            min_recall=None,
            budget=None,
            delta=config.delta_query,
    ):
        self.query_type = query_type
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.delta = delta
        self.budget = budget
