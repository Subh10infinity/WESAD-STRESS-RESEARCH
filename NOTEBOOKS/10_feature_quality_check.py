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
    / "wesad_hrv_feature_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC DATASET INFORMATION")
print("=" * 80)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nSubjects:")
print(df["subject"].nunique())

print("\nSubject IDs:")
print(sorted(df["subject"].unique()))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("2. CLASS DISTRIBUTION")
print("=" * 80)

class_counts = df["label"].value_counts().sort_index()

print(class_counts)

print("\nPercentages:")

class_percentages = (
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for label, percentage in class_percentages.items():

    class_name = (
        "Non-stress"
        if label == 0
        else "Stress"
    )

    print(
        f"{class_name:<12}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# WINDOWS PER SUBJECT
# ============================================================

print("\n" + "=" * 80)
print("3. WINDOWS PER SUBJECT")
print("=" * 80)

subject_counts = (
    df.groupby("subject")
    .size()
    .sort_index()
)

print(subject_counts)

print(
    f"\nMinimum windows per subject: "
    f"{subject_counts.min()}"
)

print(
    f"Maximum windows per subject: "
    f"{subject_counts.max()}"
)


# ============================================================
# CLASS DISTRIBUTION BY SUBJECT
# ============================================================

print("\n" + "=" * 80)
print("4. CLASS DISTRIBUTION BY SUBJECT")
print("=" * 80)

subject_class = pd.crosstab(
    df["subject"],
    df["label"]
)

subject_class = subject_class.rename(
    columns={
        0: "Non-stress",
        1: "Stress"
    }
)

print(subject_class)


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_columns = [
    "mean_hr",
    "mean_rr",
    "sdnn",
    "rmssd",
    "num_peaks",
    "invalid_rr_percentage",
    "eda_mean",
    "eda_std",
    "temp_mean",
    "temp_std",
    "acc_mean",
    "acc_std"
]


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("5. MISSING VALUE CHECK")
print("=" * 80)

missing = df[feature_columns].isna().sum()

print(missing)

total_missing = missing.sum()

print(
    f"\nTotal missing feature values: "
    f"{total_missing}"
)


# ============================================================
# INFINITE VALUES
# ============================================================

print("\n" + "=" * 80)
print("6. INFINITE VALUE CHECK")
print("=" * 80)

infinite_counts = np.isinf(
    df[feature_columns].to_numpy()
).sum(axis=0)

infinite_series = pd.Series(
    infinite_counts,
    index=feature_columns
)

print(infinite_series)

print(
    f"\nTotal infinite values: "
    f"{infinite_series.sum()}"
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("7. FEATURE STATISTICS")
print("=" * 80)

statistics = df[feature_columns].describe().T

print(
    statistics[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max"
        ]
    ].to_string()
)


# ============================================================
# CONSTANT / NEAR-CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 80)
print("8. CONSTANT / NEAR-CONSTANT FEATURES")
print("=" * 80)

for feature in feature_columns:

    unique_values = df[feature].nunique()

    std = df[feature].std()

    print(
        f"{feature:<28} "
        f"unique={unique_values:<6} "
        f"std={std:.8f}"
    )


# ============================================================
# CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("9. FEATURE CORRELATIONS")
print("=" * 80)

correlation = df[
    feature_columns
].corr()

print(
    correlation.round(3).to_string()
)


# ============================================================
# HIGH CORRELATION PAIRS
# ============================================================

print("\n" + "=" * 80)
print("10. HIGHLY CORRELATED FEATURE PAIRS")
print("=" * 80)

threshold = 0.90

found_pairs = False

for i in range(len(feature_columns)):

    for j in range(i + 1, len(feature_columns)):

        feature_1 = feature_columns[i]
        feature_2 = feature_columns[j]

        corr_value = correlation.loc[
            feature_1,
            feature_2
        ]

        if abs(corr_value) >= threshold:

            print(
                f"{feature_1:<28} "
                f"{feature_2:<28} "
                f"correlation={corr_value:.3f}"
            )

            found_pairs = True


if not found_pairs:

    print(
        "No feature pairs exceeded "
        f"|correlation| >= {threshold}."
    )


# ============================================================
# STRESS VS NON-STRESS FEATURE MEANS
# ============================================================

print("\n" + "=" * 80)
print("11. FEATURE MEANS BY CLASS")
print("=" * 80)

class_means = (
    df.groupby("label")[feature_columns]
    .mean()
    .T
)

class_means = class_means.rename(
    columns={
        0: "Non-stress",
        1: "Stress"
    }
)

print(
    class_means.round(4).to_string()
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FEATURE QUALITY CHECK COMPLETE")
print("=" * 80)

print(
    "\nNo preprocessing or scaling has been performed "
    "in this script."
)

print(
    "Scaling will be performed later inside each "
    "LOSO training fold to prevent data leakage."
)