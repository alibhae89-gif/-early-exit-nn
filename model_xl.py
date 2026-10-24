import torch
import torch.nn as nn

WIDTH = 1024


def make_deep(w=WIDTH):
    return nn.Sequential(
        nn.Linear(64, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(),
        nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 64), nn.ReLU())


class EarlyExitNetXL(nn.Module):
    def __init__(self, w=WIDTH):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(20, 64), nn.ReLU())
        self.exit1 = nn.Linear(64, 4)
        self.deep = make_deep(w)
        self.exit2 = nn.Linear(64, 4)

    def forward(self, x):
        h = self.stem(x)
        return self.exit1(h), self.exit2(self.deep(h))

    @torch.no_grad()
    def predict(self, x, t):
        h = self.stem(x)
        conf, pred = torch.softmax(self.exit1(h), 1).max(1)
        exited = conf >= t
        hard = (~exited).nonzero(as_tuple=True)[0]
        if len(hard):
            pred[hard] = self.exit2(self.deep(h[hard])).argmax(1)
        return pred, exited


class BaselineNetXL(nn.Module):
    def __init__(self, w=WIDTH):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(20, 64), nn.ReLU())
        self.deep = make_deep(w)
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        return self.head(self.deep(self.stem(x)))
