import pandas as pd

from .validate_input import validate_and_fix
from .decision_experiments import run_experiments


def sensitivity_table(results: pd.DataFrame) -> pd.DataFrame:
    df = results.sort_values("threshold").reset_index(drop=True)

    df["delta_threshold"] = df["threshold"].diff()
    df["delta_approval"] = df["approval_rate"].diff()
    df["delta_impact"] = df["total_impact"].diff()
    df["delta_accuracy"] = df["accuracy"].diff()

    # "Cost per +1 approval" (how much impact you sacrifice to gain approvals)
    # Negative delta_impact means impact got worse.
    df["impact_cost_per_approval"] = df["delta_impact"] / df["delta_approval"]

    return df


def highlight_cliffs(df: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    temp = df.copy()

    # Bigger "cliff" = more negative delta_impact for a small gain in approval
    # We'll rank by absolute cost per approval (ignore first row NaNs)
    temp = temp.dropna(subset=["impact_cost_per_approval"])
    temp["cliff_score"] = temp["impact_cost_per_approval"].abs()

    return temp.sort_values("cliff_score", ascending=False).head(top_k)


if __name__ == "__main__":
    path = "data/raw/sample_data.csv"

    raw = pd.read_csv(path)
    data = validate_and_fix(raw)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = run_experiments(data, thresholds)

    table = sensitivity_table(results)

    print("\nPolicy sensitivity\n")
    cols = [
        "threshold",
        "approval_rate",
        "total_impact",
        "accuracy",
        "delta_approval",
        "delta_impact",
        "impact_cost_per_approval",
    ]
    print(table[cols].to_string(index=False))

    cliffs = highlight_cliffs(table, top_k=2)
    if not cliffs.empty:
        print("\nBiggest cliff zones\n")
        for _, r in cliffs.iterrows():
            print(
                f"Move to {r['threshold']:.1f}: approval {r['delta_approval']:+.2f}, "
                f"impact {r['delta_impact']:+.1f}, cost/approval {r['impact_cost_per_approval']:.1f}"
            )
