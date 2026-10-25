
import pandas as pd
from sklearn.model_selection import train_test_split


def load_splits(seed=42):
    df = pd.read_csv("sms/sms.tsv", sep="\t", header=None, names=["label", "text"])
    y = (df.label == "spam").astype(int).values
    X = df.text.values
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=seed)
    return Xtr, ytr, Xva, yva, Xte, yte
