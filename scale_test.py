import copy, time
import numpy as np
import torch
import torch.nn as nn

d = np.load("data.npz")
Xtr, ytr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"]).long()
Xva, yva = torch.tensor(d["X_val"]), torch.tensor(d["y_val"]).long()
Xte, yte = torch.tensor(d["X_test"]), torch.tensor(d["y_test"]).long()
ce = nn.CrossEntropyLoss()


def make_deep(w):
    return nn.Sequential(
        nn.Linear(64, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(),
        nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 64), nn.ReLU())


class EE(nn.Module):
    def __init__(self, w):
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
        hard = (conf < t).nonzero(as_tuple=True)[0]
        if len(hard):
            pred[hard] = self.exit2(self.deep(h[hard])).argmax(1)
        return pred


class BL(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(20, 64), nn.ReLU())
        self.deep = make_deep(w)
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        return self.head(self.deep(self.stem(x)))


def train(model, early, epochs=30, batch=256):
    torch.set_num_threads(2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, state = 0.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            if early:
                l1, l2 = model(Xtr[idx])
                loss = 0.5 * ce(l1, ytr[idx]) + 0.5 * ce(l2, ytr[idx])
            else:
                loss = ce(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xva)
            acc = ((out[1] if early else out).argmax(1) == yva).float().mean().item()
        if acc > best:
            best, state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    model.eval()
    return model


xs = [Xte[i:i + 1] for i in range(500)]


def ms(fn):
    for x in xs[:100]:
        fn(x)
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        for x in xs:
            fn(x)
        runs.append((time.perf_counter() - t0) / len(xs) * 1000)
    return float(np.median(runs))


for w in [512, 1024]:
    print(f"\n=== deep width {w} (training, please wait) ===", flush=True)
    torch.manual_seed(42); bl = train(BL(w), early=False)
    torch.manual_seed(42); ee = train(EE(w), early=True)
    torch.set_num_threads(1)
    with torch.no_grad():
        bl_acc = (bl(Xte).argmax(1) == yte).float().mean().item()
        bl_ms = ms(lambda x: bl(x).argmax(1))
        print(f"baseline: test acc {bl_acc:.3f}, {bl_ms:.3f} ms/sample", flush=True)
        for t in [0.6, 0.8, 0.9]:
            acc = (ee.predict(Xte, t) == yte).float().mean().item()
            conf = torch.softmax(ee.exit1(ee.stem(Xte)), 1).max(1).values
            rate = (conf >= t).float().mean().item()
            m = ms(lambda x: ee.predict(x, t))
            print(f"  t={t}: test acc {acc:.3f}, exit rate {rate:.3f}, "
                  f"{m:.3f} ms/sample, speedup {bl_ms / m:.2f}x", flush=True)
