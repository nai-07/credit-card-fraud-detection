from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

FEATURES = [
    "amount", "hour", "merchant_category", "distance_km",
    "transactions_last_hour", "device_new", "international"
]
NUMERIC_FEATURES = [
    "amount", "hour", "distance_km", "transactions_last_hour",
    "device_new", "international"
]
CATEGORICAL_FEATURES = ["merchant_category"]
TARGET = "actual_fraud"
MERCHANTS = ["Groceries", "Dining", "Fuel", "Travel", "Electronics",
             "Online Shopping", "Healthcare", "Entertainment", "Other"]


def generate_demo_transactions(n=2500, seed=42):
    """Create clearly synthetic labelled data for an offline, reproducible demo."""
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(mean=5.8, sigma=1.05, size=n), 10, 250000)
    hour = rng.integers(0, 24, size=n)
    merchant = rng.choice(MERCHANTS, size=n, p=[.18,.13,.10,.07,.12,.17,.08,.07,.08])
    distance = np.clip(rng.gamma(shape=1.5, scale=24, size=n), 0, 1500)
    velocity = np.clip(rng.poisson(lam=1.15, size=n), 0, 12)
    device_new = rng.binomial(1, .13, size=n)
    international = rng.binomial(1, .08, size=n)

    # Labels are generated from transparent risk patterns plus random noise.
    # This is intentionally synthetic, not a model of actual bank fraud.
    logit = (
        -5.0
        + 0.000018 * amount
        + 0.045 * np.maximum(distance - 60, 0)
        + 0.42 * velocity
        + 1.55 * device_new
        + 1.35 * international
        + 1.15 * ((hour <= 4) | (hour >= 23))
        + 1.1 * np.isin(merchant, ["Electronics", "Online Shopping", "Travel"])
        + 0.75 * (amount > 25000)
        + 0.65 * ((distance > 250) & (international == 1))
    )
    probability = 1 / (1 + np.exp(-np.clip(logit, -15, 15)))
    labels = rng.binomial(1, probability)
    # Ensure both classes are represented in small sample sizes.
    if n >= 20 and labels.sum() == 0:
        labels[np.argmax(probability)] = 1
    if n >= 20 and labels.sum() == n:
        labels[np.argmin(probability)] = 0

    return pd.DataFrame({
        "transaction_id": [f"TXN-{i+1:06d}" for i in range(n)],
        "amount": np.round(amount, 2),
        "hour": hour.astype(int),
        "merchant_category": merchant,
        "distance_km": np.round(distance, 2),
        "transactions_last_hour": velocity.astype(int),
        "device_new": device_new.astype(int),
        "international": international.astype(int),
        TARGET: labels.astype(int),
    })


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("merchant", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if len(np.unique(y_test)) > 1 else None,
        "pr_auc": float(average_precision_score(y_test, probabilities)) if len(np.unique(y_test)) > 1 else None,
    }
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }


def train_and_save_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = generate_demo_transactions(5000, seed=42)
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    evaluation = evaluate_model(model, X_test, y_test)
    bundle = {
        "model": model,
        "evaluation": evaluation,
        "model_name": "Random Forest",
        "features": FEATURES,
        "data_note": "Trained on synthetic demonstration data.",
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train_model(force_retrain=False):
    if MODEL_PATH.exists() and not force_retrain:
        try:
            bundle = joblib.load(MODEL_PATH)
            if "model" in bundle and "evaluation" in bundle:
                return bundle
        except Exception:
            pass
    return train_and_save_model()


def score_transactions(model, transactions):
    """Validate input, generate fraud probabilities and add review labels."""
    if not isinstance(transactions, pd.DataFrame):
        transactions = pd.DataFrame(transactions)
    missing = [col for col in FEATURES if col not in transactions.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    result = transactions.copy()
    for col in NUMERIC_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if result[NUMERIC_FEATURES].isna().any().any():
        bad = result[NUMERIC_FEATURES].columns[result[NUMERIC_FEATURES].isna().any()].tolist()
        raise ValueError("Numeric fields contain missing or non-numeric values: " + ", ".join(bad))
    if (result["amount"] < 0).any():
        raise ValueError("Transaction amount cannot be negative.")
    if (~result["hour"].between(0, 23)).any():
        raise ValueError("Hour must be between 0 and 23.")
    if (result["distance_km"] < 0).any() or (result["transactions_last_hour"] < 0).any():
        raise ValueError("Distance and transaction count cannot be negative.")
    for col in ["device_new", "international"]:
        if not result[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 or 1.")
    if result["merchant_category"].isna().any():
        raise ValueError("Merchant category cannot be empty.")

    probabilities = model.predict_proba(result[FEATURES])[:, 1]
    result["fraud_probability"] = probabilities.round(6)
    result["prediction"] = (probabilities >= 0.5).astype(int)
    result["risk_band"] = pd.cut(
        probabilities, bins=[-0.001, 0.30, 0.70, 1.001],
        labels=["Low", "Review", "High"]
    ).astype(str)
    return result
