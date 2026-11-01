import pandas as pd
import pytest

from backend.fraud_model import FEATURES, build_pipeline, generate_demo_transactions, score_transactions


@pytest.fixture(scope="module")
def fitted_model():
    data = generate_demo_transactions(1000, seed=123)
    model = build_pipeline()
    model.fit(data[FEATURES], data["actual_fraud"])
    return model


def valid_transaction():
    return pd.DataFrame([{
        "amount": 1200.0,
        "hour": 14,
        "merchant_category": "Groceries",
        "distance_km": 4.0,
        "transactions_last_hour": 1,
        "device_new": 0,
        "international": 0,
    }])


def test_prediction_output_has_probability_and_label(fitted_model):
    result = score_transactions(fitted_model, valid_transaction())
    assert "fraud_probability" in result.columns
    assert "prediction" in result.columns
    assert 0.0 <= result.loc[0, "fraud_probability"] <= 1.0
    assert result.loc[0, "prediction"] in (0, 1)


def test_missing_column_raises_readable_error(fitted_model):
    with pytest.raises(ValueError, match="Missing required columns"):
        score_transactions(fitted_model, pd.DataFrame([{"amount": 100}]))


def test_invalid_hour_is_rejected(fitted_model):
    row = valid_transaction()
    row.loc[0, "hour"] = 27
    with pytest.raises(ValueError, match="Hour must be between"):
        score_transactions(fitted_model, row)


def test_negative_amount_is_rejected(fitted_model):
    row = valid_transaction()
    row.loc[0, "amount"] = -20
    with pytest.raises(ValueError, match="cannot be negative"):
        score_transactions(fitted_model, row)
