from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fraud_model import generate_demo_transactions

out = PROJECT_ROOT / "data" / "demo_transactions.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df = generate_demo_transactions(5000, seed=42)
df.to_csv(out, index=False)
print(f"Generated {len(df)} synthetic transactions at {out}")
print(df["actual_fraud"].value_counts().rename(index={0:"legitimate",1:"fraud"}))
