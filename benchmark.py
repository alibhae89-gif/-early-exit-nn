
import json
import time
import numpy as np
import torch
from model import EarlyExitNet, BaselineNet

torch.set_num_threads(1)  # steadier timings

d = np.load("data.npz")
Xva, yva = torch.tensor(d["X_val"]), torch.tensor(d["y_val"]).long()
Xte, yte = torch.tensor(d["X_test"]), torch.tensor(d["y_test"]).long()

ee = EarlyExitNet(); ee.load_state_dict(torch.load("early_exit.pt")); ee.eval()
bl = BaselineNet(); bl.load_state_dict(torch.load("baseline.pt")); bl.eval()

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
N, REPEATS = 1000, 7
xs = [Xte[i:i + 1] for i in range(N)]


def ms_per_sample(fn):
    for x in xs[:200]:          # warm-up
        fn(x)
    runs = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for x in xs:
            fn(x)
        runs.append((time.perf_counter() - t0) / N * 1000)
    return float(np.median(runs))


with torch.no_grad():
    base_val = (bl(Xva).argmax(1) == yva).float().mean().item()
    base_test = (bl(Xte).argmax(1) == yte).float().mean().item()
    base_ms = ms_per_sample(lambda x: bl(x).argmax(1))

    rows = []
    for t in THRESHOLDS:
        pv, _ = ee.predict(Xva, t)
        pt, ex = ee.predict(Xte, t)
        ms = ms_per_sample(lambda x: ee.predict(x, t))
        rows.append({
            "threshold": t,
            "val_acc": (pv == yva).float().mean().item(),
            "test_acc": (pt == yte).float().mean().item(),
            "exit_rate": ex.float().mean().item(),
            "ms": ms,
            "speedup": base_ms / ms,
        })

print(f"BASELINE (full depth): test acc {base_test:.3f}, {base_ms:.3f} ms/sample\n")
print("threshold  val_acc  test_acc  exit_rate  ms/sample  speedup")
for r in rows:
    print(f"{r['threshold']:>9}  {r['val_acc']:.3f}    {r['test_acc']:.3f}     "
          f"{r['exit_rate']:.3f}      {r['ms']:.3f}     {r['speedup']:.2f}x")

ok = [r for r in rows if r["val_acc"] >= base_val - 0.01]
rec = min(ok, key=lambda r: r["threshold"]) if ok else None
if rec:
    print(f"\nRECOMMENDED threshold (chosen on validation): {rec['threshold']}")
    print(f"  test acc {rec['test_acc']:.3f} vs baseline {base_test:.3f}, "
          f"exit rate {rec['exit_rate']:.1%}, speedup {rec['speedup']:.2f}x")
else:
    print("\nNo threshold stayed within 1 point of the baseline.")

json.dump({"baseline": {"test_acc": base_test, "ms": base_ms},
           "rows": rows, "recommended": rec}, open("results.json", "w"), indent=2)
print("\nSaved results.json")
