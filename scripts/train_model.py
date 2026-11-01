from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fraud_model import train_and_save_model

if __name__ == "__main__":
    bundle = train_and_save_model()
    print("Model training complete.")
    print(f"Saved model: models/fraud_model.joblib")
    print("Evaluation metrics:")
    for key, value in bundle["evaluation"]["metrics"].items():
        print(f"  {key}: {value}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(bundle["evaluation"]["confusion_matrix"])
