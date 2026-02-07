# Decision Systems Lab
# Simple decision inspection dashboard

import io
import numpy as np
import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = ["risk_score", "amount", "outcome"]


def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(content))


def load_default_data() -> pd.DataFrame:
    return pd.read_csv("data/raw/sample_data.csv")


def compute_decision(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()

    df["decision"] = (df["risk_score"] < threshold).astype(int)

    df["impact"] = np.where(
        (df["decision"] == 1) & (df["outcome"] == 1),
        df["amount"] * 0.1,
        np.where(
            (df["decision"] == 1) & (df["outcome"] == 0),
            -df["amount"] * 0.2,
            0
        )
    )

    return df


def get_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    approved = int((df["decision"] == 1).sum())
    rejected = int((df["decision"] == 0).sum())
    approval_rate = approved / total if total else 0

    accuracy = float((df["decision"] == df["outcome"]).mean()) if total else 0

    total_impact = float(df["impact"].sum()) if total else 0
    avg_impact = float(df["impact"].mean()) if total else 0

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": approval_rate,
        "accuracy": accuracy,
        "total_impact": total_impact,
        "avg_impact": avg_impact
    }


def main():
    st.set_page_config(page_title="Decision Systems Lab", layout="wide")
    st.title("Decision Systems Lab")

    st.write(
        "This dashboard shows how a decision rule behaves.\n"
        "Upload a CSV or use the sample file, then adjust the threshold."
    )

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        st.caption("Required columns: risk_score, amount, outcome")

        threshold = st.slider("Risk threshold", 0.05, 0.95, 0.60, 0.05)

    if uploaded_file is not None:
        df_raw = load_csv_from_upload(uploaded_file)
        source_label = "Uploaded file"
    else:
        df_raw = load_default_data()
        source_label = "data/raw/sample_data.csv"

    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df = compute_decision(df_raw, threshold)
    m = get_metrics(df)

    st.caption(f"Source: {source_label}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", m["total"])
    c2.metric("Approved", m["approved"])
    c3.metric("Approval rate", f'{m["approval_rate"]:.2%}')
    c4.metric("Accuracy", f'{m["accuracy"]:.2%}')
    c5.metric("Total impact", f'{m["total_impact"]:.2f}')

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Preview")
        st.dataframe(df.head(25), use_container_width=True)

    with right:
        st.subheader("Breakdown")
        approved_good = int(((df["decision"] == 1) & (df["outcome"] == 1)).sum())
        approved_bad = int(((df["decision"] == 1) & (df["outcome"] == 0)).sum())
        rejected_good = int(((df["decision"] == 0) & (df["outcome"] == 1)).sum())
        rejected_bad = int(((df["decision"] == 0) & (df["outcome"] == 0)).sum())

        st.write(f"Approved + good: {approved_good}")
        st.write(f"Approved + bad: {approved_bad}")
        st.write(f"Rejected + good: {rejected_good}")
        st.write(f"Rejected + bad: {rejected_bad}")
        st.write(f"Average impact: {m['avg_impact']:.2f}")

    st.subheader("Impact distribution")
    hist_values, hist_edges = np.histogram(df["impact"], bins=20)
    hist_df = pd.DataFrame({
        "bin_start": hist_edges[:-1],
        "count": hist_values
    }).set_index("bin_start")
    st.bar_chart(hist_df)

    st.subheader("Cumulative impact (sorted by risk)")
    df_line = df.sort_values("risk_score")[["risk_score", "impact"]].copy()
    df_line["cumulative_impact"] = df_line["impact"].cumsum()
    st.line_chart(df_line.set_index("risk_score")["cumulative_impact"])


if __name__ == "__main__":
    main()
