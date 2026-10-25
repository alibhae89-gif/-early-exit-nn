import json
import time

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sms_data import load_splits

torch.set_num_threads(2)
Xtr, ytr, Xva, yva, Xte, yte = load_splits()

s1 = joblib.load("stage1.joblib")
vec, clf = s1["vec"], s1["clf"]
tok = AutoTokenizer.from_pretrained("stage2")
model = AutoModelForSequenceClassification.from_pretrained("stage2").eval()


def stage1_batch(X):
    p = clf.predict_proba(vec.transform([str(t) for t in X]))
    return p.max(1), p.argmax(1)


def stage2_batch(X, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            enc = tok([str(t) for t in X[i:i + bs]], padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
            out.append(model(**enc).logits.argmax(1))
    return torch.cat(out).numpy()


def run_stage1(text):
    p = clf.predict_proba(vec.transform([text]))[0]
    return float(p.max()), int(p.argmax())


def run_stage2(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        return int(model(**enc).logits.argmax(1))


def cascade(text, t):
    conf, pred = run_stage1(text)
    return pred if conf >= t else run_stage2(text)


sample = [str(t) for t in Xte[:300]]


def ms_per_message(fn, repeats=3):
    for t in sample[:15]:
        fn(t)
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for t in sample:
            fn(t)
        runs.append((time.perf_counter() - t0) / len(sample) * 1000)
    return float(np.median(runs))


def acc(p, y):
    return float((p == y).mean())


def spam_stats(p, y):
    recall = float(((p == 1) & (y == 1)).sum() / (y == 1).sum())
    false_alarms = int(((p == 1) & (y == 0)).sum())
    return recall, false_alarms


print("Computing accuracies...", flush=True)
c_va, p1_va = stage1_batch(Xva)
c_te, p1_te = stage1_batch(Xte)
p2_va = stage2_batch(Xva)
p2_te = stage2_batch(Xte)

print("Timing (about 3 minutes)...", flush=True)
heavy_ms = ms_per_message(run_stage2)
cheap_ms = ms_per_message(run_stage1)

rows = []
for t in [0.7, 0.8, 0.9, 0.95, 0.99]:
    pv = np.where(c_va >= t, p1_va, p2_va)
    pt = np.where(c_te >= t, p1_te, p2_te)
    ms = ms_per_message(lambda x: cascade(x, t))
    recall, fa = spam_stats(pt, yte)
    rows.append({"threshold": t, "val_acc": acc(pv, yva), "test_acc": acc(pt, yte),
                 "exit_rate": float((c_te >= t).mean()), "ms": ms,
                 "speedup": heavy_ms / ms, "hours_per_million": ms / 3.6,
                 "spam_recall": recall, "false_alarms": fa})

r1, f1 = spam_stats(p1_te, yte)
r2, f2 = spam_stats(p2_te, yte)
print(f"\nCHEAP ONLY  : test acc {acc(p1_te, yte):.3f}, {cheap_ms:.2f} ms, spam caught {r1:.1%}, false alarms {f1}")
print(f"HEAVY ONLY  : test acc {acc(p2_te, yte):.3f}, {heavy_ms:.1f} ms, spam caught {r2:.1%}, false alarms {f2}")
print(f"  (heavy only = {heavy_ms / 3.6:.1f} server-hours per million messages)\n")
print("threshold  test_acc  answered_by_cheap  ms/msg  speedup  hrs/million  spam_caught  false_alarms")
for r in rows:
    print(f"{r['threshold']:>9}  {r['test_acc']:.3f}     {r['exit_rate']:.1%}            "
          f"{r['ms']:.2f}    {r['speedup']:.1f}x     {r['hours_per_million']:.2f}       "
          f"{r['spam_recall']:.1%}       {r['false_alarms']}")

base_val = acc(p2_va, yva)
ok = [r for r in rows if r["val_acc"] >= base_val - 0.005]
rec = min(ok, key=lambda r: r["threshold"]) if ok else rows[-1]
print(f"\nCHOSEN threshold (picked on validation): {rec['threshold']}")
json.dump({"cheap": {"test_acc": acc(p1_te, yte), "ms": cheap_ms},
           "heavy": {"test_acc": acc(p2_te, yte), "ms": heavy_ms},
           "rows": rows, "chosen": rec}, open("results_sms.json", "w"), indent=2)
print("Saved results_sms.json")
