import numpy as np
import torch
import torch.nn as nn
from model import BaselineNet

torch.manual_seed(42)
torch.set_num_threads(2)

d = np.load("data.npz")
Xtr, ytr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"]).long()
Xva, yva = torch.tensor(d["X_val"]), torch.tensor(d["y_val"]).long()

model = BaselineNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
EPOCHS, BATCH = 60, 256

best = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr))
    total = 0.0
    for i in range(0, len(Xtr), BATCH):
        idx = perm[i:i + BATCH]
        loss = ce(model(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)

    model.eval()
    with torch.no_grad():
        acc = (model(Xva).argmax(1) == yva).float().mean().item()
    if acc > best:
        best = acc
        torch.save(model.state_dict(), "baseline.pt")
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch {epoch:2d}  loss {total/len(Xtr):.3f}  val acc {acc:.3f}")

print(f"best baseline val acc: {best:.3f}  (saved to baseline.pt)")
