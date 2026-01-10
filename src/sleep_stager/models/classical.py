"""Classical ML baselines implemented with scikit-learn models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(slots=True)
class ClassicalConfig:
    model_type: str = "logreg"
    max_iter: int = 200
    C: float = 1.0
    svm_kernel: str = "rbf"
    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42


def build_model(config: ClassicalConfig) -> Any:
    model_type = config.model_type.lower()
    if model_type == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=config.max_iter,
                        C=config.C,
                        multi_class="multinomial",
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
    if model_type == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=config.C,
                        kernel=config.svm_kernel,
                        probability=True,
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            class_weight="balanced",
            random_state=config.random_state,
        )
    raise ValueError(f"Unsupported classical model_type: {config.model_type}")


def build_pipeline(config: ClassicalConfig) -> Any:
    return build_model(config)


def train_model(X: np.ndarray, y: np.ndarray, config: ClassicalConfig) -> Any:
    model = build_model(config)
    model.fit(X, y)
    return model


def predict(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict_proba(X)
    preds = model.classes_[np.argmax(probs, axis=1)]
    return preds, probs
