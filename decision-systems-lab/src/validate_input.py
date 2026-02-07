import pandas as pd


REQUIRED_COLUMNS = ["risk_score", "amount", "outcome"]


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def validate_and_fix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Basic types
    for c in ["risk_score", "amount", "outcome"]:
        if not _is_numeric_series(df[c]):
            raise ValueError(f"Column '{c}' must be numeric.")

    # Outcome should be 0/1 (or close)
    unique_outcomes = sorted(df["outcome"].dropna().unique().tolist())
    if not set(unique_outcomes).issubset({0, 1}):
        raise ValueError("Column 'outcome' must contain only 0 and 1 values.")

    # Risk score range check + fix
    rs = df["risk_score"].astype(float)
    rs_min, rs_max = float(rs.min()), float(rs.max())

    # If it already looks like 0..1, keep it.
    if 0 <= rs_min and rs_max <= 1:
        return df

    # If it looks like 0..100, normalize to 0..1
    if 0 <= rs_min and rs_max <= 100:
        df["risk_score"] = rs / 100.0
        return df

    # If it’s something else, do a safe min-max normalization
    # (keeps ordering, squeezes into 0..1)
    if rs_max > rs_min:
        df["risk_score"] = (rs - rs_min) / (rs_max - rs_min)
        return df

    raise ValueError("risk_score has no variation (all values are the same).")


if __name__ == "__main__":
    path = "data/raw/sample_data.csv"
    df = pd.read_csv(path)

    cleaned = validate_and_fix(df)

    print("Input looks good.")
    print(f"Rows: {len(cleaned)}")
    print(f"risk_score range: {cleaned['risk_score'].min():.4f} to {cleaned['risk_score'].max():.4f}")
