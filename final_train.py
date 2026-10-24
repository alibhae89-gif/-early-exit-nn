import copy, json, time
import numpy as np
import torch
import torch.nn as nn
from model_xl import EarlyExitNetXL, BaselineNetXL, WIDTH

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
d = np.load("data.npz")
Xtr, ytr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"]).long()
Xva, yva = torch.tensor(d["X_val"]), torch.tensor(d["y_val"]).long()
Xte, yte = torch.tensor(d["X_test"]), torch.tensor(d["y_test"]).long()
ce = nn.CrossEntropyLoss()


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


print("Training baseline (width 1024)...", flush=True)
torch.manual_seed(42); bl = train(BaselineNetXL(), early=False)
print("Training early-exit model (width 1024)...", flush=True)
torch.manual_seed(42); ee = train(EarlyExitNetXL(), early=True)
torch.save(bl.state_dict(), "bl_xl.pt")
torch.save(ee.state_dict(), "ee_xl.pt")

torch.set_num_threads(1)
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


with torch.no_grad():
    bl_val = (bl(Xva).argmax(1) == yva).float().mean().item()
    bl_test = (bl(Xte).argmax(1) == yte).float().mean().item()
    bl_ms = ms(lambda x: bl(x).argmax(1))
    rows = []
    for t in THRESHOLDS:
        pv, _ = ee.predict(Xva, t)
        pt, ex = ee.predict(Xte, t)
        m = ms(lambda x: ee.predict(x, t))
        rows.append({"threshold": t,
                     "val_acc": (pv == yva).float().mean().item(),
                     "test_acc": (pt == yte).float().mean().item(),
                     "exit_rate": ex.float().mean().item(),
                     "ms": m, "speedup": bl_ms / m})

print(f"\nBASELINE: test acc {bl_test:.3f}, {bl_ms:.3f} ms/sample\n")
print("threshold  val_acc  test_acc  exit_rate  ms/sample  speedup")
for r in rows:
    print(f"{r['threshold']:>9}  {r['val_acc']:.3f}    {r['test_acc']:.3f}     "
          f"{r['exit_rate']:.3f}      {r['ms']:.3f}     {r['speedup']:.2f}x")

ok = [r for r in rows if r["val_acc"] >= bl_val - 0.01]
rec = min(ok, key=lambda r: r["threshold"]) if ok else rows[-1]
print(f"\nCHOSEN threshold (picked on validation): {rec['threshold']}")
print(f"  test acc {rec['test_acc']:.3f} vs baseline {bl_test:.3f}, "
      f"exit rate {rec['exit_rate']:.1%}, speedup {rec['speedup']:.2f}x")

json.dump({"threshold": rec["threshold"], "width": WIDTH}, open("config.json", "w"))
json.dump({"baseline": {"test_acc": bl_test, "ms": bl_ms}, "rows": rows,
           "chosen": rec}, open("results_xl.json", "w"), indent=2)
print("\nSaved ee_xl.pt, bl_xl.pt, config.json, results_xl.json")
