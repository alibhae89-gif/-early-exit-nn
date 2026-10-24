import torch
import torch.nn as nn


class EarlyExitNet(nn.Module):
    def __init__(self, in_dim=20, n_classes=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        self.exit1 = nn.Linear(64, n_classes)
        self.deep = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.exit2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # Training mode: return both exits' predictions
        h = self.stem(x)
        return self.exit1(h), self.exit2(self.deep(h))

    @torch.no_grad()
    def predict(self, x, threshold):
        # Inference mode: exit early when confident, else go deeper
        h = self.stem(x)
        probs = torch.softmax(self.exit1(h), dim=1)
        conf, pred = probs.max(dim=1)
        exited = conf >= threshold
        hard = (~exited).nonzero(as_tuple=True)[0]
        if len(hard) > 0:
            pred[hard] = self.exit2(self.deep(h[hard])).argmax(dim=1)
        return pred, exited

class BaselineNet(nn.Module):
    def __init__(self, in_dim=20, n_classes=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        self.deep = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.deep(self.stem(x)))
