import pandas as pd

def explain_policy(row):
    explanation = []

    explanation.append(
        f"Threshold {row['threshold']:.2f} approves "
        f"{row['approval_rate']*100:.1f}% of cases."
    )

    explanation.append(
        f"Accuracy is {row['accuracy']*100:.1f}%."
    )

    if row['total_impact'] < 0:
        explanation.append(
            f"Total impact is negative ({row['total_impact']:.1f}), "
            "indicating net loss."
        )
    else:
        explanation.append(
            f"Total impact is positive ({row['total_impact']:.1f}), "
            "indicating net gain."
        )

    explanation.append(
        f"Average impact per decision: {row['avg_impact']:.2f}."
    )

    return " ".join(explanation)


if __name__ == "__main__":
    # Load scenario results produced earlier
    df = pd.read_csv("data/results/scenario_results.csv")

    print("\nPolicy explanations:\n")

    for _, row in df.iterrows():
        print(f"- {explain_policy(row)}\n")
