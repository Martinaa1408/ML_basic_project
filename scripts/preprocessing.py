
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def apply_safe_smote(X, y):
    min_class_size = min(Counter(y).values())
    safe_k = max(1, min(min_class_size - 1, 5))
    sm = SMOTE(random_state=42, k_neighbors=safe_k)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    return X_resampled, y_resampled

def binarize_labels(y, class_labels):
    y_bin = []
    for true_label in y:
        y_bin.append([1 if true_label == c else 0 for c in class_labels])
    return np.array(y_bin)
