# 📊Decision Systems Lab 

A decision-policy dashboard that converts **risk scores** into **defensible operating policies**.  

The system helps decision-makers choose thresholds that balance **throughput**, **accuracy**, and **business impact**, using transparent and auditable logic.
This project explores how data, analytics, and AI are used together
to design and evaluate decision-making systems.

---

## 👤Author

**Amulya Naga Raj**  
M.S. Computer Science, Syracuse University  

**Tech Stack:**  
Python • Pandas • NumPy • Streamlit • Matplotlib • CSV Analytics • Data Analysis


[![Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://decision-systems-lab.streamlit.app)
[![Live](https://img.shields.io/badge/status-online-success)](https://decision-systems-lab.streamlit.app)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---
**Click the **Demo** badge to access the live interactive dashboard.**
---
## Project Motivation

In many real systems (credit approval, fraud screening, admissions, eligibility checks), machine-learning models output **scores**, not decisions.

This project answers a critical question:

> **At what threshold should we operate the model to maximize business value while managing risk?**

Instead of focusing on model training, this dashboard focuses on:
- **Decision governance**
- **Policy explainability**
- **Operational tradeoffs**
- **Business-aligned metrics**

---

## What the Dashboard Does

The dashboard:
1. Accepts a CSV dataset containing **risk scores** and **ground-truth labels**
2. Applies a **threshold-based decision policy**
3. Evaluates outcomes using **business impact rules**
4. Sweeps multiple thresholds
5. Recommends the best operating policy based on:
   - Business objective
   - Minimum approval constraint
6. Produces **auditable artifacts** (CSV + text summaries)

---

## Decision Logic

### Decision Rule

- If risk_score < threshold → APPROVE
- Else → REJECT


### Label Semantics
- `0` → Good outcome
- `1` → Bad outcome

### Decision Outcomes
| Decision | Label | Meaning |
|--------|-------|--------|
| Approve + Good | approve_good | Correct approval |
| Approve + Bad | approve_bad | Risk exposure |
| Reject + Good | reject_good | Missed opportunity |
| Reject + Bad | reject_bad | Correct rejection |

Each outcome has a **business impact value** that can be customized.

---

## CSV Upload Formats 

The dashboard supports **two upload patterns**.

---

**OPTION A: Test with RAW DATASET CSV**

Each row represents **one decision instance**.

---
**OPTION B**

## Required Columns

### 1. Risk / Probability Column (MANDATORY)

Must represent a **probability or risk score between 0 and 1**.

**Recommended column name:**
- risk_score


**Accepted alternatives:**
- score
- probability
- prob
- pred_proba
- model_score
- risk
- p_bad


**Rules:**
- Values must be numeric
- Values outside `[0, 1]` are clipped
- Non-numeric values are dropped

---

### 2. Label Column (MANDATORY)

Indicates the **true outcome**.

**Recommended column name:**
- label


**Accepted alternatives:**
- target
- y
- outcome
- is_bad
- bad
- fraud
- default
- class


**Accepted values:**
- Numeric: `0`, `1`
- Boolean: `true / false`
- Text: `good / bad`, `yes / no`

**Interpretation:**
- 0 → good
- 1 → bad


---

## Optional Columns

These are **not required**, but enable filtering and richer analysis.

### Segment / Group Column
Used for segment-wise filtering.

**Recommended:**
- segment


**Accepted:**
- customer_segment
- group
- bucket
- cohort

---

### Date Column
Used for time-based filtering.

**Recommended:**
- date


**Accepted:**
- dt
- timestamp
- created_at
- event_date


**Accepted formats:**
- YYYY-MM-DD
- MM/DD/YYYY


---

### Amount / Value Column
Used for value-based filtering.

**Recommended:**
- amount


**Accepted:**
- value
- transaction_amount
- txn_amount
- revenue


---

## Example Raw Dataset (Correct Format)

```csv
risk_score	label	segment	     date	    amount
0.12	      0	    Consumer	 1/15/2025 	45.23
0.89	      1	    Enterprise	 1/16/2025	892.5
0.34	      0	    SMB	         1/17/2025	156.78
0.67	      1	    Consumer	 1/18/2025  234.9
0.23	      0	    Enterprise	 1/19/2025  678.45
0.78	      1	    SMB	         1/20/2025  345.67
0.45	      0	    Consumer	 1/21/2025  123.45
0.91	      1	    Enterprise	 1/22/2025	1234.56
0.15	      0	    SMB	         1/23/2025	89.12
```

## Large File Support 

The dashboard includes automatic large-file handling.
1. Smart Mode Behavior:
- Detects large CSVs
- Switches to streaming computation
- Reads data in chunks
- Prevents memory crashes

2. Manual Controls:
- Force streaming mode
- Disable smart mode if needed

3. What Works in Streaming Mode
- Full threshold sweep
- Policy recommendation
- Sensitivity analysis
- Exports

4. What Uses Preview Data Only
- Filters
- Score histogram

5. Business Scenarios
- Each scenario defines impact values per outcome.

**Examples:**
- Baseline
- Growth
- Fraud-heavy
- Conservative

**modify if needed:**
- approve_good
- approve_bad
- reject_good
- reject_bad

This allows stakeholders to stress-test policies under different assumptions.


**This project demonstrates:**
- Decision systems thinking
- Business-aware analytics
- Explainability over black-box ML
- Production-safe data handling
- Stakeholder-ready outputs

It intentionally focuses on policy evaluation, not model training.

**How to Run Locally:**
- pip install -r requirements.txt
- streamlit run dashboard.py

**Open:**
http://localhost:8501

**OUTPUT:**
Uploading a CSV with risk_score and label produces a complete decision-policy analysis: recommended threshold, business impact curves, sensitivity analysis, and auditable exports.
