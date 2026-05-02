"""# 6. Classification Model

Trains an SVM with RBF kernel using GridSearchCV and cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC


@dataclass
class ClassifierArtifacts:
    model: GridSearchCV
    metrics: dict[str, object]
    x_test: object
    y_test: object


def train_svm_classifier(x_scaled, y) -> ClassifierArtifacts:
    """Train and evaluate an optimized RBF-SVM classifier."""
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = {
        "C": [0.5, 1, 5, 10],
        "gamma": ["scale", 0.05, 0.1, 0.5],
        "kernel": ["rbf"],
    }
    search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid=grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    y_pred = search.predict(x_test)
    labels = list(search.best_estimator_.classes_)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
    }
    return ClassifierArtifacts(model=search, metrics=metrics, x_test=x_test, y_test=y_test)


def predict_risk(model: GridSearchCV, row_scaled) -> dict[str, object]:
    """Predict risk class and confidence for a single scaled feature row."""
    probabilities = model.predict_proba(row_scaled)[0]
    classes = model.best_estimator_.classes_
    idx = int(np.argmax(probabilities))
    return {
        "label": str(classes[idx]),
        "confidence": float(probabilities[idx]),
        "probabilities": {str(label): float(prob) for label, prob in zip(classes, probabilities)},
    }
