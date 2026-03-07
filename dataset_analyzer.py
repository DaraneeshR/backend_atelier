"""
dataset_analyzer.py
────────────────────────────────────────────────────────
Loads and preprocesses the real SR APPAREL SHOP dataset.
Derives:
  - Gender-specific anthropometric ratios (chest/shoulder, sleeve/arm)
  - Gender-specific industry ease averages
  - Gender-specific percentile size thresholds (XS‥XXL)

Output is consumed by gender_config.py and size_classifier.py.
"""

import os
import pandas as pd
import numpy as np

# ── Dataset config ──────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "SR APPAREL SHOP - Form Responses 1.csv"
)

# Columns to keep (upper-body only)
UPPER_BODY_COLS = {
    "ENTER THE GENDER": "gender",
    "FULL CHEST":        "chest",
    "SHOULDER":          "shoulder",
    "SLEEVE LENGTH":     "sleeve_length",
    "FRONT LENGTH ":     "front_length",   # trailing space preserved from CSV
    "ARM HOLE ":         "arm_hole",       # trailing space preserved
    "COLLAR":            "collar",
}

# PII / irrelevant columns to drop (by partial name match)
DROP_PARTIAL = ["timestamp", "name", "school", "phone", "order", "qty",
                "pant", "short", "skirt", "waist", "seat", "thigh",
                "knee", "leg open", "side length", "inseam", "shorts"]

SIZE_LABELS   = ["XS", "S", "M", "L", "XL", "XXL"]
SIZE_PERCENTILES = [0, 15, 35, 55, 75, 90, 100]   # 6 buckets from 6 boundaries


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, select upper-body cols, drop PII, drop bad rows."""
    # Strip col names
    df.columns = [c.strip() for c in df.columns]

    # Build mapping from stripped name to friendly name
    col_map = {}
    for raw, friendly in UPPER_BODY_COLS.items():
        stripped = raw.strip()
        # find match in df.columns
        for col in df.columns:
            if col.strip().upper() == stripped.upper():
                col_map[col] = friendly
                break

    # Keep only mapped columns
    available = [c for c in df.columns if c in col_map]
    df = df[available].rename(columns=col_map)

    # Clean gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.upper()
        df = df[df["gender"].isin(["MALE", "FEMALE"])]
        df["gender"] = df["gender"].str.capitalize()  # 'Male' / 'Female'

    # Convert measurement columns to numeric
    meas_cols = [c for c in df.columns if c != "gender"]
    for col in meas_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing critical fields
    critical = [c for c in ["chest", "shoulder", "sleeve_length"] if c in df.columns]
    df = df.dropna(subset=critical)

    # Drop extreme outliers per numeric column (keep within 1st–99th percentile)
    for col in meas_cols:
        if col in df.columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    return df.reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    """Return cleaned upper-body dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Please ensure 'SR APPAREL SHOP - Form Responses 1.csv' is in the project root."
        )
    df = pd.read_csv(DATASET_PATH)
    return _clean_df(df)


def derive_gender_ratios(df: pd.DataFrame) -> dict:
    """
    Derive dataset-grounded anthropometric ratios per gender.
    chest_ratio   = mean(chest / shoulder)       — used to estimate chest from shoulder
    sleeve_ratio  = mean(sleeve_length / chest)  — used to cross-validate sleeve ease
    """
    ratios = {}
    for gender in ["Male", "Female"]:
        g = df[df["gender"] == gender].copy()
        if len(g) < 10:
            continue
        chest_ratio  = (g["chest"] / g["shoulder"]).median() if "shoulder" in g else 1.9
        sleeve_ratio = (g["sleeve_length"] / g["chest"]).median() if "sleeve_length" in g else 0.72
        front_ratio  = (g["front_length"] / g["chest"]).median() if "front_length" in g else 0.52

        ratios[gender] = {
            "chest_ratio":  round(float(chest_ratio),  4),
            "sleeve_ratio": round(float(sleeve_ratio), 4),
            "front_ratio":  round(float(front_ratio),  4),
        }
    return ratios


def derive_ease_averages(df: pd.DataFrame) -> dict:
    """
    Derive gender-specific ease averages from real measurements.
    Ease = difference between the dataset distribution Q75 and median —
    representing the industry buffer built into recorded garment measurements.
    """
    ease = {}
    for gender in ["Male", "Female"]:
        g = df[df["gender"] == gender]
        if len(g) < 10:
            continue

        def ease_val(col: str) -> float:
            if col not in g.columns:
                return 4.0
            q50 = g[col].quantile(0.50)
            q75 = g[col].quantile(0.75)
            return round(float(q75 - q50), 2)

        ease[gender] = {
            "ease_chest":  max(ease_val("chest"),  3.0),
            "ease_sleeve": max(ease_val("sleeve_length"), 1.5),
            "ease_front":  max(ease_val("front_length"),  1.5),
        }
    return ease


def derive_size_thresholds(df: pd.DataFrame) -> dict:
    """
    Derive gender-specific chest-based size thresholds (XS–XXL)
    using percentile boundaries of the real dataset.
    Returns: { 'Male': {'XS': (lo, hi), 'S': (lo, hi), ...}, 'Female': {...} }
    """
    thresholds = {}
    for gender in ["Male", "Female"]:
        g = df[df["gender"] == gender]
        if len(g) < 10 or "chest" not in g.columns:
            continue

        boundaries = [float(g["chest"].quantile(p / 100)) for p in SIZE_PERCENTILES]
        thresholds[gender] = {}
        for i, label in enumerate(SIZE_LABELS):
            thresholds[gender][label] = (
                round(boundaries[i],     1),
                round(boundaries[i + 1], 1)
            )
    return thresholds


# Singleton cache so downstream modules only parse CSV once
_cache: dict | None = None


def get_analysis() -> dict:
    """
    Returns cached analysis dict:
    {
      'df':         pd.DataFrame,
      'ratios':     { 'Male': {...}, 'Female': {...} },
      'ease':       { 'Male': {...}, 'Female': {...} },
      'thresholds': { 'Male': {'XS': (lo,hi), ...}, 'Female': {...} },
    }
    """
    global _cache
    if _cache is None:
        df = load_dataset()
        _cache = {
            "df":         df,
            "ratios":     derive_gender_ratios(df),
            "ease":        derive_ease_averages(df),
            "thresholds":  derive_size_thresholds(df),
        }
    return _cache


# ── CLI quick-check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    a = get_analysis()
    df = a["df"]
    print(f"Clean dataset: {df.shape[0]} records  |  columns: {df.columns.tolist()}")
    print(f"Gender dist  : {df['gender'].value_counts().to_dict()}")
    print("\nRatios:")
    for g, v in a["ratios"].items():
        print(f"  {g}: {v}")
    print("\nEase averages:")
    for g, v in a["ease"].items():
        print(f"  {g}: {v}")
    print("\nSize thresholds (chest cm):")
    for g, sizes in a["thresholds"].items():
        print(f"  {g}:")
        for label, (lo, hi) in sizes.items():
            print(f"    {label}: {lo} – {hi} cm")
