from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "wesad_clean_feature_dataset.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH)


# ============================================================
# FEATURES WE WILL USE FOR ML
# ============================================================

feature_groups = {

    "heart": [
        "mean_hr",
        "sdnn",
        "rmssd"
    ],

    "eda": [
        "eda_mean",
        "eda_std"
    ],

    "temperature": [
        "temp_mean",
        "temp_std"
    ],

    "accelerometer": [
        "acc_mean",
        "acc_std"
    ]
}


all_features = []

for features in feature_groups.values():
    all_features.extend(features)


# ============================================================
# BASIC CHECK
# ============================================================

print("=" * 80)
print("FINAL DATASET CHECK")
print("=" * 80)

print(f"Rows    : {len(df)}")
print(f"Subjects: {df['subject'].nunique()}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# MISSING / INFINITE
# ============================================================

print("\n" + "=" * 80)
print("MISSING / INFINITE VALUES")
print("=" * 80)

missing = df[all_features].isna().sum()

infinite = np.isinf(
    df[all_features].to_numpy()
).sum(axis=0)

print("\nMissing:")
print(missing)

print("\nInfinite:")
print(
    pd.Series(
        infinite,
        index=all_features
    )
)


# ============================================================
# FEATURE RANGES
# ============================================================

print("\n" + "=" * 80)
print("FINAL FEATURE RANGES")
print("=" * 80)

for feature in all_features:

    print(
        f"{feature:<28}"
        f"min={df[feature].min():.6f}    "
        f"max={df[feature].max():.6f}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print("\nPercentages:")

print(
    (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("SUBJECT DISTRIBUTION")
print("=" * 80)

print(
    df.groupby("subject")
    .size()
    .sort_index()
)


# ============================================================
# CLASS DISTRIBUTION BY SUBJECT
# ============================================================

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION BY SUBJECT")
print("=" * 80)

print(
    pd.crosstab(
        df["subject"],
        df["label"]
    ).rename(
        columns={
            0: "Non-stress",
            1: "Stress"
        }
    )
)


# ============================================================
# FINAL FEATURE GROUPS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE CONFIGURATIONS")
print("=" * 80)

print("\nConfiguration 1 - Heart/HRV:")
print(feature_groups["heart"])

print("\nConfiguration 2 - EDA:")
print(feature_groups["eda"])

print("\nConfiguration 3 - Heart + EDA:")
print(
    feature_groups["heart"]
    + feature_groups["eda"]
)

print("\nConfiguration 4 - Full multimodal:")
print(all_features)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("FINAL CHECK COMPLETE")
print("=" * 80)

if (
    missing.sum() == 0
    and infinite.sum() == 0
    and df["subject"].nunique() == 15
):

    print(
        "Dataset is ready for the LOSO experiment."
    )

else:

    print(
        "Dataset requires further investigation."
    )