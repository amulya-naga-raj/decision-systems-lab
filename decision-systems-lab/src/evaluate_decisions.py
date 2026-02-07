# Decision Systems Lab
# Decision evaluation script

import pandas as pd


def evaluate():
    df = pd.read_csv("data/raw/sample_data.csv")

    total_rows = len(df)

    approved = int((df["decision"] == 1).sum())
    rejected = int((df["decision"] == 0).sum())
    approval_rate = approved / total_rows if total_rows else 0

    accuracy = float((df["decision"] == df["outcome"]).mean())

    total_impact = float(df["impact"].sum())
    avg_impact = float(df["impact"].mean())

    print("\nDecision evaluation summary")
    print(f"Total records        : {total_rows}")
    print(f"Approved             : {approved}")
    print(f"Rejected             : {rejected}")
    print(f"Approval rate        : {approval_rate:.2%}")
    print(f"Decision accuracy    : {accuracy:.2%}")
    print(f"Total impact         : {total_impact:.2f}")
    print(f"Average impact       : {avg_impact:.2f}")

    approved_good = int(((df["decision"] == 1) & (df["outcome"] == 1)).sum())
    approved_bad = int(((df["decision"] == 1) & (df["outcome"] == 0)).sum())
    rejected_good = int(((df["decision"] == 0) & (df["outcome"] == 1)).sum())
    rejected_bad = int(((df["decision"] == 0) & (df["outcome"] == 0)).sum())

    print("\nDecision outcome breakdown")
    print(f"Approved and good    : {approved_good}")
    print(f"Approved and bad     : {approved_bad}")
    print(f"Rejected and good    : {rejected_good}")
    print(f"Rejected and bad     : {rejected_bad}")
    print("")


if __name__ == "__main__":
    evaluate()
