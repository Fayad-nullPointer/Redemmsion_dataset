"""
Feature Engineering Script - Hospital Readmission Dataset
==========================================================
Goal: Reduce 66 columns intelligently before modelling.

Reduction plan (66 → ~30 columns):
┌─────────────────────────────────────────────────────────────────┐
│ Action                                   │ Cols removed          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Drop zero-variance cols               │ -3  (examide,         │
│    (examide, citoglipton, troglitazone)  │  citoglipton,         │
│                                          │  troglitazone)        │
├─────────────────────────────────────────────────────────────────┤
│ 2. Drop near-zero medication cols        │ -5  (acetohexamide,   │
│    (<0.1% non-zero rows)                 │  tolbutamide,         │
│                                          │  glipizide-metformin, │
│                                          │  glimepiride-         │
│                                          │  pioglitazone,        │
│                                          │  metformin-           │
│                                          │  rosiglitazone,       │
│                                          │  metformin-           │
│                                          │  pioglitazone)        │
├─────────────────────────────────────────────────────────────────┤
│ 3. Collapse 18 payer_code OHE cols       │ -17 (keep 1 combined) │
│    → 1 ordinal "payer_type" col          │                       │
├─────────────────────────────────────────────────────────────────┤
│ 4. Collapse 6 race OHE cols              │ -5  (keep 1 combined) │
│    → 1 label-encoded "race" col          │                       │
├─────────────────────────────────────────────────────────────────┤
│ 5. Engineer: total_prior_visits          │ -3 → +1               │
│    from outpatient+emergency+inpatient   │                       │
├─────────────────────────────────────────────────────────────────┤
│ 6. Engineer: total_medication_count      │  +1 (new signal)      │
│    sum of all active medication cols     │                       │
├─────────────────────────────────────────────────────────────────┤
│ 7. Engineer: any_medication_change       │  +1 (new signal)      │
│    1 if any med has Up/Down (val > 1)    │                       │
└─────────────────────────────────────────────────────────────────┘
Net result: 66 → ~34 columns
"""

import pandas as pd
import numpy as np


# ── constants ─────────────────────────────────────────────────────────────────

# Cols with a single unique value → zero information
ZERO_VAR_COLS = ["examide", "citoglipton", "troglitazone"]

# Cols where >99.9% of rows are 0 → negligible signal
NEAR_ZERO_COLS = [
    "acetohexamide", "tolbutamide", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Payer code OHE columns → collapse to ordinal groups
# Logic: group by coverage type (public > managed > private > other/unknown)
PAYER_CODE_MAP = {
    "payer_code_MC": 4,   # Medicare        (largest public)
    "payer_code_MD": 4,   # Medicaid        (public)
    "payer_code_CM": 3,   # Managed care
    "payer_code_HM": 3,   # HMO
    "payer_code_BC": 2,   # Blue Cross      (private)
    "payer_code_SP": 2,   # Self-pay        (private)
    "payer_code_CP": 2,   # Commercial plan (private)
    "payer_code_UN": 2,   # United          (private)
    "payer_code_OG": 1,   # Other gov
    "payer_code_MP": 1,   # Military plan
    "payer_code_CH": 1,   # Champus
    "payer_code_WC": 1,   # Workers comp
    "payer_code_DM": 1,   # Supplemental
    "payer_code_PO": 1,   # Point-of-service
    "payer_code_SI": 1,   # Special program
    "payer_code_OT": 0,   # Other
    "payer_code_FR": 0,   # Free
    "payer_code_Unknown": 0,
}

# Race OHE columns → collapse back to label int
RACE_MAP = {
    "race_Caucasian":        0,
    "race_AfricanAmerican":  1,
    "race_Hispanic":         2,
    "race_Asian":            3,
    "race_Other":            4,
    "race_?":                5,
}

# All medication columns (used for aggregate features)
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "tolazamide", "insulin", "glyburide-metformin",
]

PRIOR_VISIT_COLS = ["number_outpatient", "number_emergency", "number_inpatient"]


# ── main function ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce and enrich features for the hospital readmission dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded dataframe (output of encode_features).

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe with reduced column count.
    """
    df = df.copy()

    # 1. Drop zero-variance columns
    drop_zero = [c for c in ZERO_VAR_COLS if c in df.columns]
    df.drop(columns=drop_zero, inplace=True)

    # 2. Drop near-zero medication columns
    drop_near = [c for c in NEAR_ZERO_COLS if c in df.columns]
    df.drop(columns=drop_near, inplace=True)

    # 3. Collapse payer_code OHE → single ordinal column
    payer_present = [c for c in PAYER_CODE_MAP if c in df.columns]
    if payer_present:
        def get_payer_type(row):
            for col, val in PAYER_CODE_MAP.items():
                if col in row.index and row[col] == 1:
                    return val
            return 0
        df["payer_type"] = df[payer_present].apply(get_payer_type, axis=1)
        df.drop(columns=payer_present, inplace=True)

    # 4. Collapse race OHE → single label column
    race_present = [c for c in RACE_MAP if c in df.columns]
    if race_present:
        def get_race(row):
            for col, val in RACE_MAP.items():
                if col in row.index and row[col] == 1:
                    return val
            return 5  # unknown
        df["race"] = df[race_present].apply(get_race, axis=1)
        df.drop(columns=race_present, inplace=True)

    # 5. Engineer: total_prior_visits (combines 3 visit cols → 1)
    present_visits = [c for c in PRIOR_VISIT_COLS if c in df.columns]
    if present_visits:
        df["total_prior_visits"] = df[present_visits].sum(axis=1)
        df.drop(columns=present_visits, inplace=True)

    # 6. Engineer: total_medications_on (count of non-zero med cols)
    med_present = [c for c in MED_COLS if c in df.columns]
    if med_present:
        df["total_medications_on"] = (df[med_present] > 0).sum(axis=1)

    # 7. Engineer: any_medication_change (1 if any med was Up/Down i.e. val > 1)
    if med_present:
        df["any_medication_change"] = (df[med_present] > 1).any(axis=1).astype(int)

    return df


# ── quick summary ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv("encoded_train.csv")
    print(f"Before: {df.shape[1]} columns")
    df_eng = engineer_features(df)
    print(f"After:  {df_eng.shape[1]} columns")
    print("\nFinal columns:")
    print(list(df_eng.columns))