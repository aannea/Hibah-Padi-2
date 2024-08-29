import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler


class SVMClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.svm_model = svm.SVC(kernel='linear', C=1.0, random_state=42)

    def fit(self, X, y):
        # Standardize features
        X = self.scaler.fit_transform(X)
        self.svm_model.fit(X, y)

    def predict(self, X):
        # Standardize features
        X = self.scaler.transform(X)
        return self.svm_model.predict(X)
