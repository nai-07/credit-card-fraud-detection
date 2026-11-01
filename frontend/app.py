from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from backend.fraud_model import (
    FEATURES, TARGET, load_or_train_model, score_transactions,
    generate_demo_transactions, evaluate_model, DATA_DIR
)

ROOT = PROJECT_ROOT
st.set_page_config(
    page_title="Sentinel | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling: dark navy analytics console with crisp accent colors.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
.stApp { background: #0b1020; color: #edf2ff; }
[data-testid="stSidebar"] { background: #10182b; border-right: 1px solid #26334d; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.03em; }
p, label, div { font-family: 'DM Sans', sans-serif; }
.hero {
  padding: 1.4rem 1.6rem; border-radius: 18px;
  background: linear-gradient(115deg, #172747 0%, #122039 56%, #172b40 100%);
  border: 1px solid #2a3c5d; margin-bottom: 1.2rem;
}
.eyebrow { color: #7dd3fc; font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.hero-title { font-size: 2.1rem; line-height: 1.15; font-weight: 700; margin: .45rem 0; color: #f7f9ff; }
.hero-sub { color: #b8c6df; font-size: .98rem; max-width: 800px; }
.metric-card {
  background: #121c31; border: 1px solid #263650; border-radius: 15px;
  padding: 1rem 1.15rem; min-height: 115px;
}
.metric-label { color: #9eacc5; font-size: .83rem; }
.metric-value { color: #f5f8ff; font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 700; margin-top: .3rem; }
.metric-note { color: #8da0bf; font-size: .75rem; margin-top: .25rem; }
.section-label { color: #91a6c8; text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
div[data-testid="stMetric"] { background: #121c31; padding: 1rem; border: 1px solid #263650; border-radius: 14px; }
div[data-testid="stMetricLabel"] { color: #a7b6d0; }
div[data-testid="stMetricValue"] { color: #f5f8ff; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: #111b30; border-radius: 10px 10px 0 0; padding: .6rem 1rem; }
div.stButton > button[kind="primary"] { background: #38bdf8; color: #07111f; border: none; font-weight: 700; border-radius: 10px; }
div.stButton > button { border-radius: 10px; }
.small-muted { color: #8da0bf; font-size: .82rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_or_train_model()

model_bundle = get_model()
model = model_bundle["model"]
evaluation = model_bundle["evaluation"]

if "transactions" not in st.session_state:
    st.session_state.transactions = score_transactions(model, generate_demo_transactions(500, seed=27))

st.sidebar.markdown("## 🛡️ SENTINEL")
st.sidebar.caption("FRAUD INTELLIGENCE CONSOLE")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Live Transaction Check", "Batch Analysis", "Model Performance"],
    label_visibility="collapsed"
)
st.sidebar.divider()
st.sidebar.markdown("**System status**")
st.sidebar.success("Model loaded")
st.sidebar.caption(f"Model: {model_bundle['model_name']}")
st.sidebar.caption("Mode: academic demonstration")
st.sidebar.divider()
st.sidebar.markdown("**Data source**")
st.sidebar.caption("Synthetic demo transactions by default. Upload a labelled CSV for batch analysis.")
st.sidebar.warning("Demo scores are not validated for real banking use.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Transaction risk monitoring · ML powered</div>
  <div class="hero-title">Fraud Intelligence Center</div>
  <div class="hero-sub">A machine-learning demonstration that scores transaction patterns, surfaces suspicious activity, and helps an analyst prioritise review.</div>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    df = st.session_state.transactions.copy()
    fraud_count = int((df["prediction"] == 1).sum()) if "prediction" in df else 0
    total = len(df)
    avg_risk = float(df["fraud_probability"].mean()) if total else 0
    high_risk = int((df["fraud_probability"] >= 0.70).sum()) if total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions analysed", f"{total:,}", "Across current session")
    m2.metric("Flagged for review", f"{fraud_count:,}", f"{(fraud_count / max(total,1))*100:.1f}% of transactions")
    m3.metric("High-risk signals", f"{high_risk:,}", "Risk score ≥ 70%")
    m4.metric("Average risk score", f"{avg_risk*100:.1f}%", "Model-estimated probability")

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("### Risk activity")
        chart_df = df.copy()
        chart_df["Risk band"] = pd.cut(
            chart_df["fraud_probability"], bins=[-0.001, 0.30, 0.70, 1.001],
            labels=["Low", "Review", "High"]
        )
        counts = chart_df["Risk band"].value_counts().reindex(["Low", "Review", "High"], fill_value=0).rename_axis("Risk band").reset_index(name="Transactions")
        fig = px.bar(counts, x="Risk band", y="Transactions", color="Risk band",
                     color_discrete_map={"Low":"#34d399","Review":"#fbbf24","High":"#fb7185"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=10,r=10,t=10,b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Recent high-risk signals")
        recent = df.sort_values("fraud_probability", ascending=False).head(6)
        for _, row in recent.iterrows():
            pct = row["fraud_probability"] * 100
            color = "#fb7185" if pct >= 70 else "#fbbf24" if pct >= 30 else "#34d399"
            st.markdown(
                f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:.72rem .1rem;border-bottom:1px solid #263650;">
                <div><b>{row.get('transaction_id','TXN')}</b><div class="small-muted">{row['merchant_category']} · ₹{row['amount']:,.2f}</div></div>
                <div style="color:{color};font-weight:700;">{pct:.1f}%</div></div>""",
                unsafe_allow_html=True
            )
    st.markdown("### Transaction feed")
    display_cols = ["transaction_id", "amount", "merchant_category", "hour", "distance_km", "transactions_last_hour", "fraud_probability", "prediction", "risk_band"]
    feed = df[display_cols].copy()
    feed["fraud_probability"] = (feed["fraud_probability"] * 100).round(1).astype(str) + "%"
    feed["prediction"] = feed["prediction"].map({0:"Review not triggered",1:"Flagged"})
    st.dataframe(feed.sort_values("fraud_probability", ascending=False), use_container_width=True, hide_index=True)

elif page == "Live Transaction Check":
    st.markdown("### Score a single transaction")
    st.markdown('<p class="small-muted">Enter transaction signals below. The model returns a risk estimate and a review recommendation.</p>', unsafe_allow_html=True)
    with st.form("transaction_form"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Transaction amount (₹)", min_value=1.0, max_value=1000000.0, value=2450.0, step=100.0)
        hour = c2.slider("Hour of day", 0, 23, 22)
        merchant_category = c3.selectbox("Merchant category", ["Groceries", "Dining", "Fuel", "Travel", "Electronics", "Online Shopping", "Healthcare", "Entertainment", "Other"])
        c4, c5, c6 = st.columns(3)
        distance_km = c4.number_input("Distance from usual location (km)", min_value=0.0, max_value=20000.0, value=8.0, step=1.0)
        transactions_last_hour = c5.number_input("Transactions in last hour", min_value=0, max_value=100, value=1, step=1)
        device_new = c6.selectbox("New / unfamiliar device?", ["No", "Yes"])
        international = st.selectbox("International transaction?", ["No", "Yes"])
        submitted = st.form_submit_button("Analyse transaction", type="primary", use_container_width=True)

    if submitted:
        one = pd.DataFrame([{
            "amount": amount, "hour": hour, "merchant_category": merchant_category,
            "distance_km": distance_km, "transactions_last_hour": transactions_last_hour,
            "device_new": 1 if device_new == "Yes" else 0,
            "international": 1 if international == "Yes" else 0
        }])
        result = score_transactions(model, one)
        p = float(result.iloc[0]["fraud_probability"])
        band = "HIGH RISK" if p >= .70 else "REVIEW REQUIRED" if p >= .30 else "LOW RISK"
        st.divider()
        a, b = st.columns([1, 1.4])
        with a:
            st.metric("Estimated fraud risk", f"{p*100:.1f}%")
            if p >= .70:
                st.error(f"🚨 {band} — prioritise manual review.")
            elif p >= .30:
                st.warning(f"⚠️ {band} — inspect transaction context.")
            else:
                st.success(f"✓ {band} — no strong risk signal in this demo model.")
        with b:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=p*100, number={"suffix":"%"},
                gauge={"axis":{"range":[0,100]}, "bar":{"color":"#38bdf8"},
                       "steps":[{"range":[0,30],"color":"#14382f"},{"range":[30,70],"color":"#493c20"},{"range":[70,100],"color":"#4a2433"}]}))
            gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=25,r=25,t=25,b=10))
            st.plotly_chart(gauge, use_container_width=True)
        st.caption("This probability is produced by a model trained on synthetic demonstration data. It is not a real-world fraud guarantee.")

elif page == "Batch Analysis":
    st.markdown("### Analyse a transaction file")
    st.markdown("Upload a CSV with these columns: `amount, hour, merchant_category, distance_km, transactions_last_hour, device_new, international`. Optional `actual_fraud` labels let you compare predictions with known outcomes.")
    uploaded = st.file_uploader("Choose transaction CSV", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            st.write("Preview", raw.head(8))
            scored = score_transactions(model, raw)
            scored["risk_band"] = pd.cut(scored["fraud_probability"], [-0.001, .30, .70, 1.001], labels=["Low", "Review", "High"]).astype(str)
            st.success(f"Scored {len(scored):,} transaction(s).")
            st.dataframe(scored, use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"), "fraud_analysis_results.csv", "text/csv", type="primary")
            if "actual_fraud" in raw.columns:
                from sklearn.metrics import classification_report, confusion_matrix
                y_true = pd.to_numeric(raw["actual_fraud"], errors="coerce")
                valid = y_true.notna()
                if valid.any():
                    y_pred = scored.loc[valid, "prediction"].astype(int)
                    st.markdown("#### Evaluation against supplied labels")
                    st.code(classification_report(y_true[valid].astype(int), y_pred, zero_division=0))
                    cm = confusion_matrix(y_true[valid].astype(int), y_pred, labels=[0,1])
                    fig = px.imshow(cm, text_auto=True, x=["Predicted legitimate","Predicted fraud"], y=["Actual legitimate","Actual fraud"], color_continuous_scale="Blues", aspect="auto")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350)
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not analyse this CSV: {exc}")
            st.info("Check that all required columns exist and values are valid. See README.md for the exact schema.")
    else:
        st.info("No file uploaded yet. You can download a sample CSV to test the batch workflow.")
        sample = generate_demo_transactions(25, seed=81)
        sample_cols = FEATURES
        sample_out = sample[sample_cols].copy()
        sample_out["actual_fraud"] = sample["actual_fraud"]
        st.download_button("Download sample transaction CSV", sample_out.to_csv(index=False).encode("utf-8"), "sample_transactions.csv", "text/csv")

elif page == "Model Performance":
    st.markdown("### Model performance")
    st.markdown('<p class="small-muted">Metrics are calculated on a held-out synthetic test set. They demonstrate the evaluation workflow, not expected performance on real bank transactions.</p>', unsafe_allow_html=True)
    m = evaluation["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{m['precision']:.3f}")
    c2.metric("Recall", f"{m['recall']:.3f}")
    c3.metric("F1 score", f"{m['f1']:.3f}")
    c4.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "N/A")
    st.markdown("#### Confusion matrix")
    cm = evaluation["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, x=["Predicted legitimate","Predicted fraud"], y=["Actual legitimate","Actual fraud"], color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Feature importance")
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig2 = px.bar(
        imp,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Teal"
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Precision and recall are both important because fraud labels are typically rare and false positives can inconvenience legitimate cardholders.")