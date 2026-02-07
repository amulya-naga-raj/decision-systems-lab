import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("data/results")
OUT_FILE = RESULTS_DIR / "scenario_comparison.csv"

SCENARIOS = {
    "baseline": "scenario_results_baseline.csv",
    "growth": "scenario_results_growth.csv",
    "fraud_heavy": "scenario_results_fraud_heavy.csv",
    "conservative": "scenario_results_conservative.csv",
}

def load_one(scenario_name: str, filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    df = pd.read_csv(path)

    # basic sanity for required columns
    required = {"threshold", "approval_rate", "accuracy", "total_impact", "avg_impact"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} missing columns: {sorted(missing)}")

    df = df.copy()
    df["scenario"] = scenario_name
    return df

def build_comparison() -> pd.DataFrame:
    frames = []
    for name, file in SCENARIOS.items():
        frames.append(load_one(name, file))
    return pd.concat(frames, ignore_index=True)

def choose_best_per_scenario(df: pd.DataFrame) -> pd.DataFrame:
    # "least loss" = maximize total_impact
    best = (
        df.sort_values(["scenario", "total_impact"], ascending=[True, False])
          .groupby("scenario", as_index=False)
          .first()
    )
    return best[["scenario", "threshold", "approval_rate", "accuracy", "total_impact", "avg_impact"]]

def print_table(title: str, df: pd.DataFrame):
    print(f"\n{title}\n" + "-" * len(title))
    print(df.to_string(index=False))

def main():
    print("Policy comparator (cross-scenario)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_comparison()

    # Save full comparison table
    df_sorted = df.sort_values(["scenario", "threshold"])
    df_sorted.to_csv(OUT_FILE, index=False)

    best = choose_best_per_scenario(df)

    print_table("Best threshold per scenario (max total impact)", best)

    # overall best among the bests
    overall = best.sort_values("total_impact", ascending=False).head(1)
    print_table("Overall best scenario+threshold", overall)

    print(f"\nSaved:\n - {OUT_FILE}")

if __name__ == "__main__":
    main()
