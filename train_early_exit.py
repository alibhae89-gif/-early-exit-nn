
import numpy as np
import torch
import torch.nn as nn
from model import EarlyExitNet

torch.manual_seed(42)
torch.set_num_threads(2)

d = np.load("data.npz")
Xtr, ytr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"]).long()
Xva, yva = torch.tensor(d["X_val"]), torch.tensor(d["y_val"]).long()

model = EarlyExitNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
W1, W2 = 0.5, 0.5
EPOCHS, BATCH = 60, 256

best = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr))
    total = 0.0
    for i in range(0, len(Xtr), BATCH):
        idx = perm[i:i + BATCH]
        l1, l2 = model(Xtr[idx])
        loss = W1 * ce(l1, ytr[idx]) + W2 * ce(l2, ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)

    model.eval()
    with torch.no_grad():
        v1, v2 = model(Xva)
        a1 = (v1.argmax(1) == yva).float().mean().item()
        a2 = (v2.argmax(1) == yva).float().mean().item()
    if a2 > best:
        best = a2
        torch.save(model.state_dict(), "early_exit.pt")
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch {epoch:2d}  loss {total/len(Xtr):.3f}  "
              f"exit-1 val acc {a1:.3f}  exit-2 val acc {a2:.3f}")

print(f"best exit-2 val acc: {best:.3f}  (saved to early_exit.pt)")
