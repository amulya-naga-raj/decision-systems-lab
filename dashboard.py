# dashboard.py by Amulya Naga Raj
# 
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# App Config
st.set_page_config(
    page_title="Decision Systems Lab Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
/* Global */
.block-container { padding-top: 1.3rem; padding-bottom: 2.5rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
small, .stCaption { opacity: 0.9; }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #070b12 100%); }
section[data-testid="stSidebar"] * { color: #e6edf6 !important; }
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stMultiSelect {
  background: rgba(255,255,255,0.05);
  border-radius: 12px;
  padding: 8px;
}

/* Cards */
.kpi-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 18px;
  padding: 14px 16px;
}
.kpi-title { font-size: 0.85rem; opacity: 0.85; }
.kpi-value { font-size: 1.55rem; font-weight: 700; margin-top: 6px; }
.kpi-sub { font-size: 0.85rem; opacity: 0.75; margin-top: 4px; }

/* Section panels */
.panel {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 18px;
  padding: 16px 16px;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.2rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# Scenario rules
DEFAULT_SCENARIOS = {
    "baseline": dict(approve_good=50.0, approve_bad=-200.0, reject_good=-20.0, reject_bad=150.0),
    "growth": dict(approve_good=70.0, approve_bad=-180.0, reject_good=-35.0, reject_bad=120.0),
    "fraud_heavy": dict(approve_good=45.0, approve_bad=-450.0, reject_good=-15.0, reject_bad=220.0),
    "conservative": dict(approve_good=40.0, approve_bad=-300.0, reject_good=-10.0, reject_bad=180.0),
}
DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


@dataclass
class DatasetContract:
    score_col: str
    label_col: str
    extra_cols: List[str]

# Helpers: Robust column detection
def _find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _looks_like_sweep(df: pd.DataFrame) -> bool:
    """Detect if the uploaded CSV is a pre-computed sweep results table."""
    need = {"threshold", "approval_rate", "accuracy", "total_impact"}
    return need.issubset(set(df.columns))


def detect_contract(df: pd.DataFrame) -> DatasetContract:
    score_candidates = [
        "risk_score", "score", "probability", "prob", "pred_proba", "prediction", "model_score", "risk", "p_bad"
    ]
    label_candidates = ["label", "target", "y", "outcome", "is_bad", "bad", "fraud", "default", "class"]

    score_col = _find_first_existing(df, score_candidates)
    label_col = _find_first_existing(df, label_candidates)

    if score_col is None or label_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        score_guess = None
        for c in numeric_cols:
            s = df[c].dropna()
            if len(s) and s.min() >= 0 and s.max() <= 1:
                score_guess = c
                break

        label_guess = None
        for c in df.columns:
            s = df[c].dropna()
            if len(s) == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                uniq = sorted(pd.unique(s).tolist())
                if len(uniq) <= 3 and all(u in [0, 1] for u in uniq):
                    label_guess = c
                    break
            else:
                uniq = set(str(x).strip().lower() for x in pd.unique(s).tolist())
                if uniq.issubset({"0", "1", "true", "false", "good", "bad", "yes", "no"}):
                    label_guess = c
                    break

        score_col = score_col or score_guess
        label_col = label_col or label_guess

    if score_col is None or label_col is None:
        raise ValueError(
            "Could not auto-detect required columns.\n"
            "Upload either:\n"
            "A) A raw dataset with a probability/score column (0..1) AND a label column (0=good, 1=bad), or\n"
            "B) A sweep results CSV containing: threshold, approval_rate, accuracy, total_impact, (avg_impact optional)."
        )

    extra_cols = [c for c in df.columns if c not in [score_col, label_col]]
    return DatasetContract(score_col=score_col, label_col=label_col, extra_cols=extra_cols)


def normalize_score_label(df: pd.DataFrame, contract: DatasetContract) -> pd.DataFrame:
    d = df.copy()

    d[contract.score_col] = pd.to_numeric(d[contract.score_col], errors="coerce")
    d = d.dropna(subset=[contract.score_col]).copy()
    d[contract.score_col] = d[contract.score_col].clip(0, 1)

    raw = d[contract.label_col]

    def to01(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            if x in [0, 1]:
                return int(x)
        s = str(x).strip().lower()
        if s in ["1", "true", "yes", "bad", "fraud", "default"]:
            return 1
        if s in ["0", "false", "no", "good"]:
            return 0
        return np.nan

    d[contract.label_col] = raw.apply(to01)
    d = d.dropna(subset=[contract.label_col]).copy()
    d[contract.label_col] = d[contract.label_col].astype(int)

    return d


def make_decisions(scores: np.ndarray, threshold: float) -> np.ndarray:
    # 1=approve if risk_score < threshold else reject
    return (scores < threshold).astype(int)


# Core policy computations (in-memory)
def compute_metrics(df: pd.DataFrame, contract: DatasetContract, threshold: float, impacts: Dict[str, float]) -> Dict[str, float]:
    scores = df[contract.score_col].to_numpy()
    y = df[contract.label_col].to_numpy()  # 0=good, 1=bad

    approve = make_decisions(scores, threshold)
    reject = 1 - approve

    approve_good = int(np.sum((approve == 1) & (y == 0)))
    approve_bad = int(np.sum((approve == 1) & (y == 1)))
    reject_good = int(np.sum((reject == 1) & (y == 0)))
    reject_bad = int(np.sum((reject == 1) & (y == 1)))

    correct = approve_good + reject_bad
    acc = correct / len(df) if len(df) else 0.0
    approval_rate = float(np.mean(approve)) if len(df) else 0.0

    total_impact = (
        approve_good * impacts["approve_good"]
        + approve_bad * impacts["approve_bad"]
        + reject_good * impacts["reject_good"]
        + reject_bad * impacts["reject_bad"]
    )
    avg_impact = total_impact / len(df) if len(df) else 0.0

    return {
        "threshold": float(threshold),
        "approval_rate": float(approval_rate),
        "accuracy": float(acc),
        "total_impact": float(total_impact),
        "avg_impact": float(avg_impact),
        "approve_good": approve_good,
        "approve_bad": approve_bad,
        "reject_good": reject_good,
        "reject_bad": reject_bad,
        "n": int(len(df)),
    }


def run_threshold_sweep(df: pd.DataFrame, contract: DatasetContract, thresholds: List[float], impacts: Dict[str, float]) -> pd.DataFrame:
    rows = [compute_metrics(df, contract, t, impacts) for t in thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_policy(sweep: pd.DataFrame, goal: str, min_approval: float) -> pd.Series:
    s = sweep.copy()
    s = s[s["approval_rate"] >= min_approval].copy()
    if s.empty:
        s = sweep.copy()

    if goal == "max_total_impact":
        return s.loc[s["total_impact"].idxmax()]
    if goal == "max_accuracy":
        return s.loc[s["accuracy"].idxmax()]
    if goal == "least_loss":
        return s.loc[s["total_impact"].idxmax()]
    return s.loc[s["total_impact"].idxmax()]


def sensitivity_table(sweep: pd.DataFrame) -> pd.DataFrame:
    s = sweep.sort_values("threshold").reset_index(drop=True).copy()
    s["delta_approval"] = s["approval_rate"].diff()
    s["delta_impact"] = s["total_impact"].diff()
    s["impact_cost_per_approval"] = s["delta_impact"] / s["delta_approval"]
    return s


# LARGE FILE MODE 
def _file_size_mb(uploaded_file) -> Optional[float]:
    try:
        return uploaded_file.size / (1024 * 1024)
    except Exception:
        return None


def streaming_sweep_csv(uploaded_file, thresholds: List[float], impacts: Dict[str, float], max_rows_preview: int = 5000):
    """
    Streaming computation over a raw dataset CSV without loading whole file in memory.
    Computes same sweep schema as in-memory mode.

    Output:
    - sweep_df
    - contract (best-effort detected on preview sample)
    - df_preview (small sample for charts + filters)
    - notes (string)
    """
    # Reset pointer for safety (Streamlit uploads can be read multiple times)
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    # Read a small sample to detect contract robustly
    preview = pd.read_csv(uploaded_file, nrows=max_rows_preview)
    if _looks_like_sweep(preview):
        # user uploaded sweep csv, not raw
        return None, None, None, "Uploaded file looks like a sweep results table (not raw rows)."

    contract = detect_contract(preview)
    preview_norm = normalize_score_label(preview, contract)

    # Prepare counters per threshold
    th = sorted(thresholds)
    counts = {
        t: {
            "approve_good": 0, "approve_bad": 0, "reject_good": 0, "reject_bad": 0, "n": 0
        } for t in th
    }

    # Rewind and iterate chunks
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    chunk_iter = pd.read_csv(uploaded_file, chunksize=200_000)
    for chunk in chunk_iter:
        # normalize using the same detected columns; if missing, fail cleanly
        if contract.score_col not in chunk.columns or contract.label_col not in chunk.columns:
            raise ValueError(
                f"Large-file mode: required columns not found in later chunks.\n"
                f"Expected: score='{contract.score_col}', label='{contract.label_col}'.\n"
                f"Tip: Ensure the whole CSV has consistent headers."
            )

        c = chunk[[contract.score_col, contract.label_col]].copy()
        c = normalize_score_label(c, contract)
        if len(c) == 0:
            continue

        scores = c[contract.score_col].to_numpy()
        y = c[contract.label_col].to_numpy()

        for t in th:
            approve = (scores < t)
            reject = ~approve

            ag = int(np.sum(approve & (y == 0)))
            ab = int(np.sum(approve & (y == 1)))
            rg = int(np.sum(reject & (y == 0)))
            rb = int(np.sum(reject & (y == 1)))

            counts[t]["approve_good"] += ag
            counts[t]["approve_bad"] += ab
            counts[t]["reject_good"] += rg
            counts[t]["reject_bad"] += rb
            counts[t]["n"] += int(len(c))

    # Build sweep df
    rows = []
    for t in th:
        n = counts[t]["n"]
        if n == 0:
            rows.append({"threshold": t, "approval_rate": 0.0, "accuracy": 0.0, "total_impact": 0.0, "avg_impact": 0.0,
                         "approve_good": 0, "approve_bad": 0, "reject_good": 0, "reject_bad": 0, "n": 0})
            continue

        ag = counts[t]["approve_good"]
        ab = counts[t]["approve_bad"]
        rg = counts[t]["reject_good"]
        rb = counts[t]["reject_bad"]

        approval_rate = (ag + ab) / n
        accuracy = (ag + rb) / n
        total_impact = (
            ag * impacts["approve_good"]
            + ab * impacts["approve_bad"]
            + rg * impacts["reject_good"]
            + rb * impacts["reject_bad"]
        )
        avg_impact = total_impact / n

        rows.append({
            "threshold": float(t),
            "approval_rate": float(approval_rate),
            "accuracy": float(accuracy),
            "total_impact": float(total_impact),
            "avg_impact": float(avg_impact),
            "approve_good": int(ag),
            "approve_bad": int(ab),
            "reject_good": int(rg),
            "reject_bad": int(rb),
            "n": int(n),
        })

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    note = "Large-file mode: computed sweep via streaming chunks (safe for big CSVs)."
    return sweep_df, contract, preview_norm, note


# Sidebar: Inputs
st.sidebar.title("Controls")

st.sidebar.subheader("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

use_sample = st.sidebar.toggle("Use sample data instead", value=(uploaded is None))

# Auto-switch toggle 
auto_switch = st.sidebar.toggle("Smart mode (auto-switch for large files)", value=True)
force_stream = st.sidebar.toggle("Force large-file mode (streaming)", value=False)

# For precomputed sweep CSVs
st.sidebar.caption("Upload types supported:")
st.sidebar.write("- Raw dataset: score/probability + label")
st.sidebar.write("- Sweep results: threshold, approval_rate, accuracy, total_impact")

st.sidebar.subheader("2) Scenario & Objective")
scenario_name = st.sidebar.selectbox("Scenario", list(DEFAULT_SCENARIOS.keys()), index=0)
goal = st.sidebar.radio(
    "Optimize for",
    ["max_total_impact", "max_accuracy", "least_loss"],
    index=0,
    help="max_total_impact: pure business value. max_accuracy: prediction correctness. least_loss: safest among losses."
)
min_approval = st.sidebar.slider("Minimum approval rate constraint", 0.0, 1.0, 0.40, 0.01)

st.sidebar.subheader("3) Policy knobs")
threshold_live = st.sidebar.slider("Live threshold (single policy view)", 0.05, 0.95, 0.30, 0.01)

st.sidebar.subheader("4) Impact rules")
imp = DEFAULT_SCENARIOS[scenario_name].copy()
imp["approve_good"] = st.sidebar.number_input("approve_good (+)", value=float(imp["approve_good"]), step=5.0)
imp["approve_bad"] = st.sidebar.number_input("approve_bad (-)", value=float(imp["approve_bad"]), step=10.0)
imp["reject_good"] = st.sidebar.number_input("reject_good (-)", value=float(imp["reject_good"]), step=5.0)
imp["reject_bad"] = st.sidebar.number_input("reject_bad (+)", value=float(imp["reject_bad"]), step=10.0)

st.sidebar.caption(
    "Tip: These represent business impact per decision outcome.\n"
    "Example: approving a bad case is costly, rejecting a bad case is beneficial."
)

st.sidebar.subheader("5) Threshold sweep")
custom_thresholds = st.sidebar.text_input(
    "Thresholds (comma-separated)",
    value=", ".join([f"{t:.2f}" for t in DEFAULT_THRESHOLDS]),
)

def parse_thresholds(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
            if 0 < v < 1:
                vals.append(v)
        except Exception:
            pass
    vals = sorted(set(vals))
    return vals if vals else DEFAULT_THRESHOLDS

thresholds = parse_thresholds(custom_thresholds)

st.sidebar.divider()
st.sidebar.caption("Dashboard link: http://localhost:8501")


# Load data 
@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    risk = rng.uniform(0, 1, size=n)
    p_bad = 0.15 + 0.7 * risk
    label = (rng.uniform(0, 1, size=n) < p_bad).astype(int)
    segments = rng.choice(["Consumer", "SMB", "Enterprise"], size=n, p=[0.45, 0.35, 0.20])
    amount = np.round(rng.lognormal(mean=4.1, sigma=0.55, size=n), 2)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
    return pd.DataFrame({"risk_score": risk, "label": label, "segment": segments, "amount": amount, "date": dates})


def load_user_csv(file) -> pd.DataFrame:
    # NOTE: don't force dtype here; keep robust; normalize later
    return pd.read_csv(file)


# Data mode selection
mode_note = ""
df_raw = None
df = None
contract = None
sweep = None
df_preview = None

if use_sample:
    df_raw = load_sample()
    contract = detect_contract(df_raw)
    df = normalize_score_label(df_raw, contract)
    df_preview = df.copy()
    sweep = run_threshold_sweep(df, contract, thresholds, imp)
    mode_note = "Mode: Sample data (in-memory)."
else:
    if uploaded is None:
        st.error("Please upload a CSV or enable sample data.")
        st.stop()

    # Peek small to classify sweep vs raw + decide mode
    try:
        uploaded.seek(0)
    except Exception:
        pass

    peek = pd.read_csv(uploaded, nrows=5000)

    if _looks_like_sweep(peek):
        # User uploaded a sweep results CSV 
        df_sweep = peek.copy()
        # If file has more than preview, read entire
        try:
            uploaded.seek(0)
        except Exception:
            pass
        df_sweep = pd.read_csv(uploaded)

        # Ensure required columns
        needed = ["threshold", "approval_rate", "accuracy", "total_impact"]
        missing = [c for c in needed if c not in df_sweep.columns]
        if missing:
            st.error(f"Sweep CSV missing columns: {missing}")
            st.stop()

        if "avg_impact" not in df_sweep.columns:
            # compute if possible (if n exists)
            if "n" in df_sweep.columns and df_sweep["n"].astype(float).replace(0, np.nan).notna().any():
                df_sweep["avg_impact"] = df_sweep["total_impact"] / df_sweep["n"].replace(0, np.nan)
                df_sweep["avg_impact"] = df_sweep["avg_impact"].fillna(0.0)
            else:
                df_sweep["avg_impact"] = np.nan

        # align schema expected downstream
        sweep = df_sweep.sort_values("threshold").reset_index(drop=True)
        mode_note = "Mode: Uploaded sweep results (no raw rows)."
        df_preview = None
        contract = None
        df = None

    else:
        # Raw dataset upload: decide in-memory vs streaming
        size_mb = _file_size_mb(uploaded)
        should_stream = False
        if force_stream:
            should_stream = True
        elif auto_switch and size_mb is not None and size_mb >= 30:
            should_stream = True

        if should_stream:
            try:
                sweep, contract, df_preview, mode_note = streaming_sweep_csv(uploaded, thresholds, imp)
            except Exception as e:
                st.error(f"Large-file mode failed: {e}")
                st.info("Tip: Ensure the CSV has consistent headers and valid score/label columns throughout.")
                st.stop()
        else:
            try:
                uploaded.seek(0)
            except Exception:
                pass
            try:
                df_raw = load_user_csv(uploaded)
                contract = detect_contract(df_raw)
                df = normalize_score_label(df_raw, contract)
                df_preview = df.copy()
                sweep = run_threshold_sweep(df, contract, thresholds, imp)
                mode_note = f"Mode: In-memory (file size ~ {size_mb:.1f} MB)." if size_mb is not None else "Mode: In-memory."
            except Exception as e:
                st.error(str(e))
                st.info("Fix: Ensure your CSV has a score/probability column in [0..1] and a binary label column (0 good, 1 bad).")
                st.stop()


# Header: Story + Purpose
st.title("Decision Systems Lab Dashboard")
st.caption(
    "A decision-policy dashboard that converts risk scores into defensible operating policies. "
    "Compare thresholds, measure business impact, and explain tradeoffs clearly."
)

# Your info + stack (no ML claims)
st.markdown(
    """
**Amulya Naga Raj**  
M.S. Computer Science, Syracuse University  
**Tech stack:** Python • Pandas • NumPy • Streamlit • Matplotlib • CSV Analytics • Data Analysis
""",
)

# Mode note
st.info(mode_note)


# FILTERS SECTION 
df_f = None
segment_col = None
date_col = None
amount_col = None

if df_preview is not None:
    df_f = df_preview.copy()
    segment_col = _find_first_existing(df_f, ["segment", "customer_segment", "group", "bucket", "cohort"])
    date_col = _find_first_existing(df_f, ["date", "dt", "timestamp", "created_at", "event_date"])
    amount_col = _find_first_existing(df_f, ["amount", "value", "transaction_amount", "txn_amount", "revenue"])

    st.subheader("Filters (optional)")
    with st.container():
        fcols = st.columns(3)

        if segment_col:
            segs = sorted(df_f[segment_col].astype(str).fillna("Unknown").unique().tolist())
            pick = fcols[0].multiselect("Segment", segs, default=segs)
            df_f = df_f[df_f[segment_col].astype(str).isin(pick)].copy()
        else:
            fcols[0].info("No segment column detected. That's fine.")

        if date_col:
            dmin = pd.to_datetime(df_f[date_col], errors="coerce").min()
            dmax = pd.to_datetime(df_f[date_col], errors="coerce").max()
            if pd.notna(dmin) and pd.notna(dmax):
                start, end = fcols[1].date_input("Date range", value=(dmin.date(), dmax.date()))
                dser = pd.to_datetime(df_f[date_col], errors="coerce")
                df_f = df_f[(dser.dt.date >= start) & (dser.dt.date <= end)].copy()
            else:
                fcols[1].info("Date column detected but not parseable as dates.")
        else:
            fcols[1].info("No date column detected. That's fine.")

        if amount_col and pd.api.types.is_numeric_dtype(df_f[amount_col]):
            amin, amax = float(df_f[amount_col].min()), float(df_f[amount_col].max())
            lo, hi = fcols[2].slider("Amount/value range", amin, amax, (amin, amax))
            df_f = df_f[(df_f[amount_col] >= lo) & (df_f[amount_col] <= hi)].copy()
        else:
            fcols[2].info("No numeric amount/value column detected. That's fine.")

    if len(df_f) < 10:
        st.warning("After filters, very few rows remain. Consider widening filters for more stable metrics.")

    # Recompute sweep on filtered data when in-memory. For streaming mode: we keep sweep from streaming, but live/filters use preview only.
    if df is not None:
        sweep = run_threshold_sweep(df_f, contract, thresholds, imp)


# Data badge / contract 
with st.container():
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.3, 1.5])

    rows_count = int(sweep["n"].max()) if ("n" in sweep.columns and sweep["n"].notna().any()) else (len(df_f) if df_f is not None else None)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Rows evaluated</div>
              <div class="kpi-value">{(rows_count if rows_count is not None else "—")}</div>
              <div class="kpi-sub">Cleaned / valid</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Scenario</div>
              <div class="kpi-value">{scenario_name}</div>
              <div class="kpi-sub">Objective: {goal}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Score column</div>
              <div class="kpi-value">{(contract.score_col if contract else "—")}</div>
              <div class="kpi-sub">Auto-detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Label column</div>
              <div class="kpi-value">{(contract.label_col if contract else "—")}</div>
              <div class="kpi-sub">0=good, 1=bad</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<hr/>", unsafe_allow_html=True)


# Core computations downstream
best = select_policy(sweep, goal=goal, min_approval=min_approval)
sens = sensitivity_table(sweep)

# Live policy metrics
if df_f is not None and contract is not None:
    live = compute_metrics(df_f, contract, threshold_live, imp)
    live_note = ""
else:
    # pick nearest threshold in sweep
    idx = (sweep["threshold"] - float(threshold_live)).abs().idxmin()
    r = sweep.loc[idx]
    live = {
        "threshold": float(r["threshold"]),
        "approval_rate": float(r["approval_rate"]),
        "accuracy": float(r["accuracy"]),
        "total_impact": float(r["total_impact"]),
        "avg_impact": float(r["avg_impact"]) if "avg_impact" in sweep.columns else float("nan"),
        "approve_good": int(r["approve_good"]) if "approve_good" in sweep.columns else 0,
        "approve_bad": int(r["approve_bad"]) if "approve_bad" in sweep.columns else 0,
        "reject_good": int(r["reject_good"]) if "reject_good" in sweep.columns else 0,
        "reject_bad": int(r["reject_bad"]) if "reject_bad" in sweep.columns else 0,
        "n": int(r["n"]) if "n" in sweep.columns else 0,
    }
    live_note = "Live view uses nearest available threshold from uploaded sweep results (no raw rows loaded)."


# Policy Flight Recorder
def build_flight_recorder(best_row: pd.Series, sweep_df: pd.DataFrame, impacts: Dict[str, float], scenario: str, objective: str, min_appr: float) -> pd.DataFrame:
    """A polished audit log stakeholders love: what was chosen, why, and what was rejected."""
    s = sweep_df.copy()
    s["meets_min_approval"] = s["approval_rate"] >= float(min_appr)
    s["rank_by_total_impact"] = s["total_impact"].rank(ascending=False, method="min").astype(int)
    s["rank_by_accuracy"] = s["accuracy"].rank(ascending=False, method="min").astype(int)

    chosen_thr = float(best_row["threshold"])
    s["chosen"] = s["threshold"].astype(float) == chosen_thr

    # add context columns
    s["scenario"] = scenario
    s["objective"] = objective
    s["min_approval"] = float(min_appr)

    # impacts snapshot
    s["impact_approve_good"] = impacts["approve_good"]
    s["impact_approve_bad"] = impacts["approve_bad"]
    s["impact_reject_good"] = impacts["reject_good"]
    s["impact_reject_bad"] = impacts["reject_bad"]

    # stakeholder-friendly reason
    def reason(row):
        if row["chosen"]:
            return "CHOSEN: best under objective + constraints"
        if not row["meets_min_approval"]:
            return "Rejected: below min approval constraint"
        return "Candidate: not the top choice for objective"

    s["decision_reason"] = s.apply(reason, axis=1)
    keep = ["scenario","objective","min_approval","threshold","approval_rate","accuracy","total_impact","avg_impact",
            "meets_min_approval","rank_by_total_impact","rank_by_accuracy","chosen","decision_reason",
            "impact_approve_good","impact_approve_bad","impact_reject_good","impact_reject_bad"]
    for col in keep:
        if col not in s.columns:
            s[col] = np.nan
    return s[keep].sort_values(["chosen","rank_by_total_impact"], ascending=[False, True]).reset_index(drop=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Scenario Explorer", "Policy Comparison", "Downloads & Narrative"])

# TAB 1
with tab1:
    st.subheader("Executive overview")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Recommended threshold</div>
              <div class="kpi-value">{best["threshold"]:.2f}</div>
              <div class="kpi-sub">Constraint: min approval {min_approval:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Approval rate (recommended)</div>
              <div class="kpi-value">{best["approval_rate"]:.0%}</div>
              <div class="kpi-sub">Throughput / growth</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Accuracy (recommended)</div>
              <div class="kpi-value">{best["accuracy"]:.0%}</div>
              <div class="kpi-sub">Correct outcomes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-title">Total impact (recommended)</div>
              <div class="kpi-value">{best["total_impact"]:.1f}</div>
              <div class="kpi-sub">Net business value</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Threshold sweep results**")
        show_cols = ["threshold", "approval_rate", "accuracy", "total_impact", "avg_impact"]
        for col in show_cols:
            if col not in sweep.columns:
                sweep[col] = np.nan
        st.dataframe(
            sweep[show_cols].style.format({
                "threshold": "{:.2f}",
                "approval_rate": "{:.0%}",
                "accuracy": "{:.0%}",
                "total_impact": "{:.1f}",
                "avg_impact": "{:.2f}",
            }),
            use_container_width=True,
            height=260
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**What this project demonstrates**")
        st.write(
            "- A score becomes a decision only after choosing a policy threshold.\n"
            "- Different business objectives produce different 'best' thresholds.\n"
            "- Impact rules translate outcomes into real value and cost.\n"
            "- The chosen policy is auditable (exportable) and defensible."
        )
        st.markdown("**Impact rules in this run**")
        st.code(
            f"approve_good={imp['approve_good']}\n"
            f"approve_bad={imp['approve_bad']}\n"
            f"reject_good={imp['reject_good']}\n"
            f"reject_bad={imp['reject_bad']}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Total impact vs threshold**")
        fig = plt.figure()
        plt.plot(sweep["threshold"], sweep["total_impact"], marker="o")
        plt.xlabel("threshold")
        plt.ylabel("total_impact")
        st.pyplot(fig, clear_figure=True)

    with cB:
        st.markdown("**Approval rate vs threshold**")
        fig = plt.figure()
        plt.plot(sweep["threshold"], sweep["approval_rate"], marker="o")
        plt.xlabel("threshold")
        plt.ylabel("approval_rate")
        st.pyplot(fig, clear_figure=True)

    cC, cD = st.columns(2)
    with cC:
        st.markdown("**Accuracy vs threshold**")
        fig = plt.figure()
        plt.plot(sweep["threshold"], sweep["accuracy"], marker="o")
        plt.xlabel("threshold")
        plt.ylabel("accuracy")
        st.pyplot(fig, clear_figure=True)

    with cD:
        st.markdown("**Average impact per decision vs threshold**")
        fig = plt.figure()
        plt.plot(sweep["threshold"], sweep["avg_impact"], marker="o")
        plt.xlabel("threshold")
        plt.ylabel("avg_impact")
        st.pyplot(fig, clear_figure=True)


# TAB 2
with tab2:
    st.subheader("Single-policy explorer (interactive)")

    if live_note:
        st.warning(live_note)

    a, b = st.columns([1.2, 0.8])
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"**Live policy @ threshold = {threshold_live:.2f}**")
        st.dataframe(
            pd.DataFrame([{
                "threshold": live["threshold"],
                "approval_rate": live["approval_rate"],
                "accuracy": live["accuracy"],
                "total_impact": live["total_impact"],
                "avg_impact": live["avg_impact"],
            }]).style.format({
                "threshold": "{:.2f}",
                "approval_rate": "{:.0%}",
                "accuracy": "{:.0%}",
                "total_impact": "{:.1f}",
                "avg_impact": "{:.2f}",
            }),
            use_container_width=True,
            height=90
        )
        st.markdown("**Decision outcome breakdown** (counts)")
        st.dataframe(
            pd.DataFrame([{
                "approve_good": live.get("approve_good", 0),
                "approve_bad": live.get("approve_bad", 0),
                "reject_good": live.get("reject_good", 0),
                "reject_bad": live.get("reject_bad", 0),
            }]),
            use_container_width=True,
            height=90
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Interpretation**")
        st.write(
            "Use this view to explain the policy to non-technical stakeholders:\n"
            "- **Approval rate** = throughput / growth.\n"
            "- **Accuracy** = correctness.\n"
            "- **Impact** = value after applying your cost rules.\n"
            "If impact is negative, the policy is destroying value even if accuracy looks decent."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if df_f is not None and contract is not None:
        st.markdown("**Score distribution (preview)**")
        fig = plt.figure()
        plt.hist(df_f[contract.score_col], bins=25)
        plt.axvline(threshold_live, linestyle="--")
        plt.xlabel(contract.score_col)
        plt.ylabel("count")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Score distribution requires raw row-level data. Upload a raw dataset CSV to enable this chart.")


# TAB 3
with tab3:
    st.subheader("Policy sensitivity and cliff zones")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Sensitivity table**")
    st.dataframe(
        sens[["threshold", "approval_rate", "total_impact", "accuracy", "delta_approval", "delta_impact", "impact_cost_per_approval"]]
        .style.format({
            "threshold": "{:.2f}",
            "approval_rate": "{:.0%}",
            "total_impact": "{:.1f}",
            "accuracy": "{:.0%}",
            "delta_approval": "{:+.0%}",
            "delta_impact": "{:+.1f}",
            "impact_cost_per_approval": "{:+.1f}",
        }),
        use_container_width=True,
        height=310
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tmp = sens.dropna(subset=["impact_cost_per_approval"]).copy()
    if not tmp.empty:
        tmp["abs_cost"] = tmp["impact_cost_per_approval"].abs()
        cliffs = tmp.sort_values("abs_cost", ascending=False).head(2)

        st.markdown("**Biggest cliff zones**")
        for _, r in cliffs.iterrows():
            st.info(
                f"Move to threshold {r['threshold']:.2f}: "
                f"approval Δ {r['delta_approval']:+.0%}, "
                f"impact Δ {r['delta_impact']:+.1f}, "
                f"impact/approval {r['impact_cost_per_approval']:+.1f}"
            )
    else:
        st.info("No sensitivity deltas available (need at least 2 thresholds).")


# TAB 4
with tab4:
    st.subheader("Exports + project narrative")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Download sweep results (CSV):**")
        csv_bytes = sweep.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download threshold sweep CSV",
            data=csv_bytes,
            file_name=f"threshold_sweep_{scenario_name}.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Download policy summary (TXT):**")
        summary = (
            f"Decision Systems Lab - Policy Summary\n"
            f"Owner: Amulya Naga Raj\n"
            f"Scenario: {scenario_name}\n"
            f"Objective: {goal}\n"
            f"Min approval: {min_approval:.0%}\n\n"
            f"Recommended threshold: {best['threshold']:.2f}\n"
            f"Approval rate: {best['approval_rate']:.0%}\n"
            f"Accuracy: {best['accuracy']:.0%}\n"
            f"Total impact: {best['total_impact']:.1f}\n"
            f"Avg impact/decision: {best['avg_impact']:.2f}\n\n"
            f"Impact rules:\n"
            f"  approve_good={imp['approve_good']}\n"
            f"  approve_bad={imp['approve_bad']}\n"
            f"  reject_good={imp['reject_good']}\n"
            f"  reject_bad={imp['reject_bad']}\n"
        )
        st.download_button(
            "Download policy summary",
            data=summary.encode("utf-8"),
            file_name=f"policy_summary_{scenario_name}.txt",
            mime="text/plain",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Policy Flight Recorder
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Policy Flight Recorder Audit Report**")
    st.caption("This is the stakeholder-friendly report that explains what was chosen and why.")
    flight = build_flight_recorder(best, sweep, imp, scenario_name, goal, min_approval)
    st.dataframe(flight, use_container_width=True, height=220)

    flight_csv = flight.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download flight recorder CSV",
        data=flight_csv,
        file_name=f"policy_flight_recorder_{scenario_name}.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("### Narrative of the project")
    st.markdown(
        f"""
**Project goal**  
Convert risk scores into a defendable decision policy by selecting a threshold that balances throughput (approvals), correctness (accuracy), and business value (impact).

**What the dashboard shows**  
1. A threshold sweep across candidate operating points ({", ".join([f"{t:.2f}" for t in thresholds])}).  
2. Business impact using scenario-specific cost rules.  
3. A recommended threshold selected by objective (**{goal}**) with a minimum approval constraint (**{min_approval:.0%}**).  
4. Sensitivity analysis highlighting "cliff zones" where small threshold changes cause large impact changes.

**Result**  
Recommended threshold = **{best["threshold"]:.2f}** with approval rate **{best["approval_rate"]:.0%}**, accuracy **{best["accuracy"]:.0%}**, and total impact **{best["total_impact"]:.1f}**.  
This supports an operational decision because it translates scores into explicit business outcomes.

**Why this is useful**  
Stakeholders can adjust cost assumptions and instantly see how the best policy changes, which mirrors how real decision systems are governed.
"""
    )

st.caption("Dashboard link: http://localhost:8501")