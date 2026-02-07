# src/scenarios.py

SCENARIOS = {
    "baseline": {
        "approve_good": 50.0,
        "approve_bad": -200.0,
        "reject_good": -20.0,
        "reject_bad": 150.0,
        "min_approval": 0.40,
        "goal": "least_loss",
    },
    "conservative": {
        "approve_good": 40.0,
        "approve_bad": -300.0,
        "reject_good": -10.0,
        "reject_bad": 180.0,
        "min_approval": 0.25,
        "goal": "least_loss",
    },
    "growth": {
        "approve_good": 70.0,
        "approve_bad": -180.0,
        "reject_good": -35.0,
        "reject_bad": 120.0,
        "min_approval": 0.55,
        "goal": "balanced",
    },
    "fraud_heavy": {
        "approve_good": 45.0,
        "approve_bad": -450.0,
        "reject_good": -15.0,
        "reject_bad": 220.0,
        "min_approval": 0.30,
        "goal": "least_loss",
    },
}
