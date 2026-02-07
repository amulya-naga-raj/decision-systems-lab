import pandas as pd

from .validate_input import validate_and_fix


def run_decision_rule(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()

    df["decision"] = (df["risk_score"] <= threshold).astype(int)

    # impact logic
    # good = outcome == 1
    # bad  = outcome == 0
    df["impact"] = 0.0

    approved = df["decision"] == 1
    good = df["outcome"] == 1
    bad = df["outcome"] == 0

    df.loc[approved & good, "impact"] = df["amount"] * 0.1
    df.loc[approved & bad, "impact"] = -df["amount"] * 0.2

    return df


def evaluate(df: pd.DataFrame) -> dict:
    total = len(df)
    approved = int(df["decision"].sum())
    approval_rate = approved / total

    correct = ((df["decision"] == 1) & (df["outcome"] == 1)) | (
        (df["decision"] == 0) & (df["outcome"] == 0)
    )
    accuracy = correct.mean()

    total_impact = df["impact"].sum()
    avg_impact = df["impact"].mean()

    return {
        "approval_rate": round(approval_rate, 2),
        "accuracy": round(accuracy, 2),
        "total_impact": round(total_impact, 1),
        "avg_impact": round(avg_impact, 3),
    }


def run_experiments(df: pd.DataFrame, thresholds):
    results = []

    for t in thresholds:
        scored = run_decision_rule(df, t)
        metrics = evaluate(scored)
        metrics["threshold"] = t
        results.append(metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":
    path = "data/raw/sample_data.csv"

    raw = pd.read_csv(path)
    df = validate_and_fix(raw)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = run_experiments(df, thresholds)

    print("\nDecision experiment results\n")
    print(results.to_string(index=False))
