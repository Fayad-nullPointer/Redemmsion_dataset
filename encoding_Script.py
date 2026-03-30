"""
Encoding Script - Hospital Readmission Dataset
================================================
Encoding strategy per column group:

1. BINARY columns (2 values)         → Label Encoding  (0/1)
   - gender, change, diabetesMed
   - all medication columns with only ['No', 'Steady']

2. MEDICATION columns (4 values)     → Ordinal Encoding
   - Values: No < Steady < Down < Up  (dosage intensity order)
   - Captures natural clinical ordering of medication dosage changes

3. AGE column                         → Ordinal Encoding
   - Age buckets have a clear natural order

4. RACE & PAYER_CODE columns          → One-Hot Encoding
   - Nominal categories with no natural order

5. DIAGNOSIS columns (diag_1/2/3)     → Frequency Encoding
   - Hundreds of ICD-9 codes → One-Hot would explode dimensionality
   - Frequency reflects how common/important each diagnosis is in the dataset

6. READMITTED (target)                → Ordinal Encoding
   - NO < >30 < <30  (urgency/risk order)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder


# ── helpers ───────────────────────────────────────────────────────────────────

MEDICATION_ORDER = [["No", "Steady", "Down", "Up"]]

MEDICATION_COLS_4VAL = [
    "metformin", "repaglinide", "nateglinide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone",
    "acarbose", "miglitol", "insulin", "glyburide-metformin",
    "chlorpropamide",  # has No/Steady/Up — still fits 4-val order
    "tolazamide",      # No/Steady/Up
]

MEDICATION_COLS_BINARY = [
    "acetohexamide", "tolbutamide", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

AGE_ORDER = [["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
              "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]]

READMITTED_ORDER = [["NO", ">30", "<30"]]  # low → high urgency

ONEHOT_COLS   = ["race", "payer_code"]
BINARY_COLS   = ["gender", "change", "diabetesMed"] + MEDICATION_COLS_BINARY
DIAG_COLS     = ["diag_1", "diag_2", "diag_3"]


# ── main function ─────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all object columns to numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (nulls handled, sparse cols dropped).

    Returns
    -------
    pd.DataFrame
        Fully numeric dataframe ready for modelling.
    """
    df = df.copy()

    # 1. Binary → Label Encoding (0 / 1)
    le = LabelEncoder()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # 2. Medication (4-value) → Ordinal Encoding
    for col in MEDICATION_COLS_4VAL:
        if col in df.columns:
            enc = OrdinalEncoder(
                categories=MEDICATION_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            df[col] = enc.fit_transform(df[[col]]).astype(int)

    # 3. Age → Ordinal Encoding
    if "age" in df.columns:
        enc = OrdinalEncoder(
            categories=AGE_ORDER,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df["age"] = enc.fit_transform(df[["age"]]).astype(int)

    # 4. Race & Payer Code → One-Hot Encoding
    present_ohe = [c for c in ONEHOT_COLS if c in df.columns]
    if present_ohe:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int)
        ohe_arr  = ohe.fit_transform(df[present_ohe])
        ohe_cols = ohe.get_feature_names_out(present_ohe)
        df = pd.concat(
            [df.drop(columns=present_ohe),
             pd.DataFrame(ohe_arr, columns=ohe_cols, index=df.index)],
            axis=1,
        )

    # 5. Diagnosis columns → Frequency Encoding
    for col in DIAG_COLS:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[col]  = df[col].map(freq_map).fillna(0)

    # 6. Target → Ordinal Encoding
    if "readmitted" in df.columns:
        enc = OrdinalEncoder(
            categories=READMITTED_ORDER,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df["readmitted"] = enc.fit_transform(df[["readmitted"]]).astype(int)

    return df