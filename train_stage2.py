import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sms_data import load_splits

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(2)
NAME, EPOCHS, BATCH, LR, MAXLEN = "distilbert-base-uncased", 2, 16, 3e-5, 64

Xtr, ytr, Xva, yva, Xte, yte = load_splits()
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2)
opt = torch.optim.AdamW(model.parameters(), lr=LR)


def batches(X, y, shuffle):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), BATCH):
        j = idx[i:i + BATCH]
        enc = tok([str(X[k]) for k in j], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt")
        yield enc, torch.tensor(y[j])


def val_acc():
    model.eval()
    ok = 0
    with torch.no_grad():
        for enc, yb in batches(Xva, yva, False):
            ok += (model(**enc).logits.argmax(1) == yb).sum().item()
    return ok / len(Xva)


steps = EPOCHS * ((len(Xtr) + BATCH - 1) // BATCH)
done, t0 = 0, time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    for enc, yb in batches(Xtr, ytr, True):
        loss = model(**enc, labels=yb).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        done += 1
        if done % 20 == 0:
            print(f"step {done}/{steps}  loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
    print(f"epoch {epoch} done, validation accuracy {val_acc():.4f}", flush=True)

model.save_pretrained("stage2")
tok.save_pretrained("stage2")
print("saved to stage2/", flush=True)
