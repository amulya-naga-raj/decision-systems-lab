import pandas as pd

from .validate_input import validate_and_fix
from .decision_experiments import run_experiments


def pick_best(results: pd.DataFrame, goal: str, min_approval: float = 0.0) -> pd.Series:
    df = results.copy()

    if min_approval > 0:
        df = df[df["approval_rate"] >= min_approval]

    if df.empty:
        raise ValueError("No thresholds match the constraints. Try lowering min_approval.")

    if goal == "max_total_impact":
        return df.sort_values(["total_impact", "accuracy"], ascending=[False, False]).iloc[0]

    if goal == "max_accuracy":
        return df.sort_values(["accuracy", "total_impact"], ascending=[False, False]).iloc[0]

    if goal == "least_loss":
        # total_impact is negative in this toy setup, so "least loss" means highest (closest to zero)
        return df.sort_values(["total_impact", "accuracy"], ascending=[False, False]).iloc[0]

    raise ValueError("Unknown goal. Use: max_total_impact, max_accuracy, least_loss")


if __name__ == "__main__":
    path = "data/raw/sample_data.csv"

    raw = pd.read_csv(path)
    df = validate_and_fix(raw)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = run_experiments(df, thresholds)

    # Change these two lines to test different “business goals”
    goal = "least_loss"       # max_total_impact | max_accuracy | least_loss
    min_approval = 0.40       # set to 0.0 if no constraint

    best = pick_best(results, goal=goal, min_approval=min_approval)

    print("\nPolicy selection\n")
    print(f"Goal: {goal}")
    print(f"Min approval: {min_approval}")
    print("\nAll results:\n")
    print(results.to_string(index=False))
    print("\nChosen threshold:\n")
    print(best.to_string())
