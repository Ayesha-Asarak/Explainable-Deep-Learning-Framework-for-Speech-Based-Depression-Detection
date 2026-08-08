"""Prototype-based few-shot classifier for participant embeddings."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class PrototypeClassifier(ClassifierMixin, BaseEstimator):
    """Classify by distance to learned class prototypes."""

    def __init__(
        self,
        metric="cosine",
        temperature=1.0,
        shrinkage=0.0,
    ):
        self.metric = metric
        self.temperature = temperature
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.array([0, 1])):
            raise ValueError("PrototypeClassifier requires binary labels 0/1")
        global_mean = X.mean(axis=0)
        prototypes = []
        for label in self.classes_:
            class_mean = X[y == label].mean(axis=0)
            prototype = (
                (1.0 - self.shrinkage) * class_mean
                + self.shrinkage * global_mean
            )
            prototypes.append(prototype)
        self.prototypes_ = np.stack(prototypes)
        return self

    @staticmethod
    def _normalize(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-12)

    def _distances(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.metric == "cosine":
            values = self._normalize(X)
            prototypes = self._normalize(self.prototypes_)
            return 1.0 - values @ prototypes.T
        if self.metric == "euclidean":
            difference = X[:, None, :] - self.prototypes_[None, :, :]
            return np.sqrt(np.sum(difference**2, axis=2))
        raise ValueError(f"Unknown prototype metric: {self.metric}")

    def predict_proba(self, X):
        logits = -self._distances(X) / max(float(self.temperature), 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)

