from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only
from collections import defaultdict
import json

class HistoryLogger(Logger):
    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)

    @property
    def name(self):
        return "history_log"

    @property
    def version(self):
        return "1.0"

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for metric_name, metric_value in metrics.items():
            self.history[metric_name].append(metric_value)
        return

    @rank_zero_only
    def save(self):
        with open('original_logging_test.json', 'w') as fp:
            json.dump(self.history, fp)

    def log_hyperparams(self, params):
        pass
