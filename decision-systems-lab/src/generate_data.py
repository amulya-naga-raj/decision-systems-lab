# Decision Systems Lab
# Sample data generator
# Plain and readable.

import pandas as pd
import numpy as np


def create_sample_data():
    # Risk score between 0 and 1
    risk_score = np.random.rand(100)

    # Amount tied to each event
    amount = np.random.randint(10, 1000, size=100)

    # Outcome: 1 = good, 0 = bad
    outcome = np.random.choice([0, 1], size=100, p=[0.4, 0.6])

    # Decision based on a simple rule
    # If risk is low, approve (1). Otherwise, reject (0).
    decision = (risk_score < 0.6).astype(int)

    # Impact of the decision
    # Approved + good → gain
    # Approved + bad → loss
    # Rejected → zero
    impact = np.where(
        (decision == 1) & (outcome == 1),
        amount * 0.1,
        np.where(
            (decision == 1) & (outcome == 0),
            -amount * 0.2,
            0
        )
    )

    data = pd.DataFrame({
        "risk_score": risk_score,
        "amount": amount,
        "outcome": outcome,
        "decision": decision,
        "impact": impact
    })

    return data


if __name__ == "__main__":
    data = create_sample_data()

    # Save the data so it can be used later
    data.to_csv("data/raw/sample_data.csv", index=False)

    print("Sample data saved to data/raw/sample_data.csv")
