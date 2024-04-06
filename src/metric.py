import torchmetrics
import torch


class LinkingAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        assert len(targets) == len(preds)

        for idx in range(len(targets)):
            if preds[idx] == targets[idx]:
                self.correct += 1

        self.total += len(targets)

    def compute(self):
        return self.correct.float() / self.total.float()

