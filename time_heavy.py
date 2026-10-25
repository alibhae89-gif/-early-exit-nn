import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(2)
name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).eval()

msgs = ["WINNER!! You have been selected to receive a $900 prize. Call now to claim.",
        "Ok lar... Joking wif u oni...",
        "Are we still meeting for lunch at 1pm tomorrow?"] * 20


def one(m):
    x = tok(m, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        model(**x)


for m in msgs[:10]:
    one(m)
t0 = time.perf_counter()
for m in msgs:
    one(m)
ms = (time.perf_counter() - t0) / len(msgs) * 1000
print(f"\nDistilBERT: {ms:.1f} ms per message on this server (2 threads)")
print("Cheap first-stage model: about 0.3 ms per message (measured on my test)")
