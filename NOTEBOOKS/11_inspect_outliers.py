from pathlib import Path

import pandas as pd


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "wesad_hrv_feature_dataset.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# TEMPERATURE STD OUTLIERS
# ============================================================

print("=" * 80)
print("TOP TEMPERATURE STD VALUES")
print("=" * 80)

print(
    df[
        [
            "subject",
            "window_id",
            "label",
            "temp_mean",
            "temp_std"
        ]
    ]
    .sort_values(
        "temp_std",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# ACCELERATION STD OUTLIERS
# ============================================================

print("\n" + "=" * 80)
print("TOP ACCELERATION STD VALUES")
print("=" * 80)

print(
    df[
        [
            "subject",
            "window_id",
            "label",
            "acc_mean",
            "acc_std"
        ]
    ]
    .sort_values(
        "acc_std",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# EDA STD OUTLIERS
# ============================================================

print("\n" + "=" * 80)
print("TOP EDA STD VALUES")
print("=" * 80)

print(
    df[
        [
            "subject",
            "window_id",
            "label",
            "eda_mean",
            "eda_std"
        ]
    ]
    .sort_values(
        "eda_std",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# HR OUTLIERS
# ============================================================

print("\n" + "=" * 80)
print("TOP HEART RATE VALUES")
print("=" * 80)

print(
    df[
        [
            "subject",
            "window_id",
            "label",
            "mean_hr",
            "sdnn",
            "rmssd"
        ]
    ]
    .sort_values(
        "mean_hr",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("OUTLIER INSPECTION COMPLETE")
print("=" * 80)