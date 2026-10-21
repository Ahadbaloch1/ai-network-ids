"""Training, evaluation, prediction and persistence."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from .data import CLASSES
from .features import FEATURE_NAMES, build_features

MODEL_VERSION = 1


def train(df, seed=42, n_estimators=150, test_size=0.2):
    """Train a supervised classifier and an unsupervised anomaly detector.

    Returns (bundle, metrics). The metrics are computed on a held-out test split.
    """
    if "label" not in df.columns:
        raise ValueError("Training data needs a 'label' column.")
    unknown = sorted(set(df["label"]) - set(CLASSES))
    if unknown:
        raise ValueError("Unknown labels: %s" % ", ".join(unknown))

    X = build_features(df)
    y = df["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                 random_state=seed, n_jobs=-1)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    clf.fit(X_train, y_train)

    # The anomaly detector learns only what normal traffic looks like.
    normal_rows = X_train[y_train == "normal"]
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=seed, n_jobs=-1)
    iso.fit(normal_rows)

    predicted = clf.predict(X_test)
    labels = [c for c in CLASSES if c in set(y)]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "report": classification_report(y_test, predicted, labels=labels, digits=3, zero_division=0),
        "labels": labels,
        "confusion": confusion_matrix(y_test, predicted, labels=labels),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    bundle = {
        "version": MODEL_VERSION,
        "classifier": clf,
        "anomaly": iso,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(clf.classes_),
    }
    return bundle, metrics


def predict(bundle, df):
    """Return a copy of df with prediction, confidence, anomaly and alert columns."""
    X = build_features(df)[bundle["feature_names"]]
    proba = bundle["classifier"].predict_proba(X)
    classes = np.array(bundle["classes"])
    out = df.copy()
    out["prediction"] = classes[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(3)
    out["anomaly"] = bundle["anomaly"].predict(X) == -1
    out["alert"] = (out["prediction"] != "normal") | out["anomaly"]
    return out


def feature_importance(bundle):
    values = bundle["classifier"].feature_importances_
    return pd.Series(values, index=bundle["feature_names"]).sort_values(ascending=False)


def save_model(bundle, path):
    joblib.dump(bundle, path)


def load_model(path):
    """Load a model file. Only load files that you trained yourself or trust."""
    bundle = joblib.load(path)
    required = {"version", "classifier", "anomaly", "feature_names", "classes"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Not a valid NIDS model file.")
    if bundle["version"] != MODEL_VERSION:
        raise ValueError("Unsupported model version: %s" % bundle["version"])
    return bundle
