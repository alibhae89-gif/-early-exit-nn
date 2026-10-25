import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sms_data import load_splits

Xtr, ytr, Xva, yva, Xte, yte = load_splits()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
clf = LogisticRegression(C=10, max_iter=1000, class_weight="balanced")
clf.fit(vec.fit_transform(Xtr), ytr)
joblib.dump({"vec": vec, "clf": clf}, "stage1.joblib")
print("train / val / test sizes:", len(Xtr), len(Xva), len(Xte))
print("stage 1 validation accuracy: %.4f" % (clf.predict(vec.transform(Xva)) == yva).mean())
