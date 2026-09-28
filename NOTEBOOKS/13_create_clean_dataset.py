from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "wesad_hrv_feature_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "wesad_clean_feature_dataset.csv"
)


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("Loading original feature dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Original windows: {len(df)}")


# ============================================================
# DOCUMENTED ARTIFACT
# ============================================================

# S3 Window 42 contains a synchronized sensor artifact:
#
# - Temperature has an extreme transient spike
# - Accelerometer has an extreme transient spike
# - EDA drops abruptly to approximately zero
#
# These occur at approximately the same time, indicating
# a likely sensor/data acquisition artifact.
#
# Therefore this window is excluded from the final ML dataset.

ARTIFACT_SUBJECT = "S3"
ARTIFACT_WINDOW = 42


# ============================================================
# REMOVE ARTIFACT
# ============================================================

artifact_mask = (
    (df["subject"] == ARTIFACT_SUBJECT)
    &
    (df["window_id"] == ARTIFACT_WINDOW)
)

print("\nArtifact found:")

print(
    df.loc[
        artifact_mask,
        [
            "subject",
            "window_id",
            "label",
            "temp_std",
            "acc_std",
            "eda_std"
        ]
    ].to_string(index=False)
)


df_clean = df.loc[
    ~artifact_mask
].copy()


# ============================================================
# RESET INDEX
# ============================================================

df_clean = df_clean.reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df_clean.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CLEAN DATASET CREATED")
print("=" * 80)

print(
    f"Original windows : {len(df)}"
)

print(
    f"Removed windows  : {len(df) - len(df_clean)}"
)

print(
    f"Final windows    : {len(df_clean)}"
)

print(
    f"Subjects         : "
    f"{df_clean['subject'].nunique()}"
)

print("\nClass distribution:")

print(
    df_clean["label"]
    .value_counts()
    .sort_index()
)

print("\nWindows per subject:")

print(
    df_clean
    .groupby("subject")
    .size()
    .sort_index()
)

print("\nSaved to:")

print(OUTPUT_PATH)