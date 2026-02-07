# src/scenario_runner.py

import os
import argparse
import pandas as pd
from src.scenarios import SCENARIOS

def apply_decision(df, threshold, impact):
    df = df.copy()
    df["decision"] = (df["risk_score"] <= threshold).astype(int)

    def calc_impact(row):
        if row["decision"] == 1 and row["outcome"] == 1:
            return impact["approve_good"]
        if row["decision"] == 1 and row["outcome"] == 0:
            return impact["approve_bad"]
        if row["decision"] == 0 and row["outcome"] == 1:
            return impact["reject_good"]
        return impact["reject_bad"]

    df["impact"] = df.apply(calc_impact, axis=1)
    return df

def metrics(df):
    total = len(df)
    approved = df["decision"].sum()
    accuracy = (df["decision"] == df["outcome"]).mean()
    total_impact = df["impact"].sum()

    return {
        "approval_rate": approved / total,
        "accuracy": accuracy,
        "total_impact": total_impact,
        "avg_impact": total_impact / total,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--input", default="data/raw/sample_data.csv")
    args = parser.parse_args()

    scenario = SCENARIOS.get(args.scenario)
    if not scenario:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    path = os.path.abspath(args.input)
    df = pd.read_csv(path)

    print("\nScenario runner")
    print(f"Scenario: {args.scenario}")
    print(f"Input: {path}")
    print(f"Rows: {len(df)}")
    print(
        "Impact rules:",
        f"approve_good={scenario['approve_good']},",
        f"approve_bad={scenario['approve_bad']},",
        f"reject_good={scenario['reject_good']},",
        f"reject_bad={scenario['reject_bad']}",
    )

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    rows = []

    print("\nThreshold sweep:")
    print("threshold  approval_rate  accuracy  total_impact  avg_impact")

    for t in thresholds:
        out = apply_decision(
            df,
            t,
            {
                "approve_good": scenario["approve_good"],
                "approve_bad": scenario["approve_bad"],
                "reject_good": scenario["reject_good"],
                "reject_bad": scenario["reject_bad"],
            },
        )
        m = metrics(out)
        rows.append({**m, "threshold": t})

        print(
            f"{t:<9} "
            f"{m['approval_rate']:<14.2f} "
            f"{m['accuracy']:<9.2f} "
            f"{m['total_impact']:<13.1f} "
            f"{m['avg_impact']:<10.1f}"
        )

    results = pd.DataFrame(rows)

    os.makedirs("data/results", exist_ok=True)
    results.to_csv("data/results/scenario_results.csv", index=False)

    print("\nSaved:")
    print(" - data/results/scenario_results.csv")

if __name__ == "__main__":
    main()
