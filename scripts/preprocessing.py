
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def apply_safe_smote(X_train, y_train, k_max=5, random_state=42):
    min_class = min(Counter(y_train).values())
    safe_k = max(1, min(min_class - 1, k_max))
    sm = SMOTE(random_state=random_state, k_neighbors=safe_k)
    return sm.fit_resample(X_train, y_train)

def binarize_labels(y, classes):
    return label_binarize(y, classes=np.unique(classes))
