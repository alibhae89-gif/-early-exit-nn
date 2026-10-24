import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

X, y = make_classification(
    n_samples=20000, n_features=20, n_informative=12, n_redundant=4,
    n_classes=4, n_clusters_per_class=2, class_sep=0.9, flip_y=0.03,
    random_state=SEED,
)

# 60% train, 20% validation, 20% test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

# Scale features using training data only
scaler = StandardScaler().fit(X_train)
X_train, X_val, X_test = (scaler.transform(a).astype("float32")
                          for a in (X_train, X_val, X_test))

np.savez("data.npz", X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

for name, a, b in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    print(name, a.shape, "class counts:", np.bincount(b).tolist())
