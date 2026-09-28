import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "results/wesad_clean_feature_dataset.csv"

OUTPUT_DIR = "results/feature_importance"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("FEATURE / SENSOR CONTRIBUTION ANALYSIS")
print("=" * 70)

print(f"\nDataset: {DATA_PATH}")
print(f"Rows: {len(df)}")
print(f"Subjects: {df['subject'].nunique()}")

print("\nClass distribution:")
print(df["label"].value_counts().sort_index())


# ============================================================
# 3. FULL MULTIMODAL FEATURES
# ============================================================

FEATURES = [
    "mean_hr",
    "sdnn",
    "rmssd",
    "eda_mean",
    "eda_std",
    "temp_mean",
    "temp_std",
    "acc_mean",
    "acc_std"
]


# ============================================================
# 4. FEATURE → SENSOR MODALITY MAPPING
# ============================================================

FEATURE_MODALITY = {

    "mean_hr": "Heart/HRV",
    "sdnn": "Heart/HRV",
    "rmssd": "Heart/HRV",

    "eda_mean": "EDA",
    "eda_std": "EDA",

    "temp_mean": "Temperature",
    "temp_std": "Temperature",

    "acc_mean": "Motion",
    "acc_std": "Motion"
}


# ============================================================
# 5. SUBJECTS
# ============================================================

subjects = sorted(df["subject"].unique())

print("\nSubjects:")
print(subjects)


# ============================================================
# 6. STORAGE
# ============================================================

feature_importance_results = []


# ============================================================
# 7. LOSO FEATURE IMPORTANCE
# ============================================================

print("\nStarting LOSO feature-importance analysis...\n")


for test_subject in subjects:

    print(f"Testing unseen subject: {test_subject}")

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_df = df[
        df["subject"] != test_subject
    ].copy()

    X_train = train_df[FEATURES].values
    y_train = train_df["label"].values

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importances = model.feature_importances_

    for feature, importance in zip(
        FEATURES,
        importances
    ):

        feature_importance_results.append({

            "test_subject": test_subject,

            "feature": feature,

            "modality": FEATURE_MODALITY[feature],

            "importance": importance
        })


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

importance_df = pd.DataFrame(
    feature_importance_results
)


# ============================================================
# 9. FEATURE-LEVEL SUMMARY
# ============================================================

feature_summary = (
    importance_df
    .groupby("feature")["importance"]
    .agg([
        "mean",
        "std",
        "min",
        "max"
    ])
    .reset_index()
)

feature_summary = feature_summary.sort_values(
    "mean",
    ascending=False
)


feature_summary_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance_summary.csv"
)

feature_summary.to_csv(
    feature_summary_path,
    index=False
)


# ============================================================
# 10. MODALITY-LEVEL SUMMARY
# ============================================================

modality_summary = (
    importance_df
    .groupby("modality")["importance"]
    .agg([
        "mean",
        "std",
        "min",
        "max"
    ])
    .reset_index()
)

modality_summary = modality_summary.sort_values(
    "mean",
    ascending=False
)


# ------------------------------------------------------------
# Normalize modality importance so that the four modality
# groups sum to approximately 1.
# ------------------------------------------------------------

modality_summary["relative_importance"] = (
    modality_summary["mean"] /
    modality_summary["mean"].sum()
)


modality_summary_path = os.path.join(
    OUTPUT_DIR,
    "modality_importance_summary.csv"
)

modality_summary.to_csv(
    modality_summary_path,
    index=False
)


# ============================================================
# 11. SUBJECT-LEVEL MODALITY IMPORTANCE
# ============================================================

subject_modality = (
    importance_df
    .groupby(
        ["test_subject", "modality"]
    )["importance"]
    .sum()
    .reset_index()
)


subject_modality_pivot = (
    subject_modality
    .pivot(
        index="test_subject",
        columns="modality",
        values="importance"
    )
    .fillna(0)
)


subject_modality_path = os.path.join(
    OUTPUT_DIR,
    "subject_modality_importance.csv"
)

subject_modality_pivot.to_csv(
    subject_modality_path
)


# ============================================================
# 12. SAVE RAW LOSO IMPORTANCES
# ============================================================

raw_path = os.path.join(
    OUTPUT_DIR,
    "loso_feature_importances.csv"
)

importance_df.to_csv(
    raw_path,
    index=False
)


# ============================================================
# 13. PRINT FEATURE RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 14. PRINT MODALITY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MODALITY IMPORTANCE")
print("=" * 70)

print(
    modality_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. PRINT SUBJECT-LEVEL MODALITY IMPORTANCE
# ============================================================

print("\n")
print("=" * 70)
print("SUBJECT-LEVEL MODALITY IMPORTANCE")
print("=" * 70)

print(
    subject_modality_pivot.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 16. SAVE SUMMARY TEXT
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance_report.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Feature / Sensor Contribution Analysis\n"
    )

    f.write(
        "=====================================\n\n"
    )

    f.write(
        "Dataset: WESAD clean 703-window dataset\n"
    )

    f.write(
        "Validation: Leave-One-Subject-Out (LOSO)\n"
    )

    f.write(
        "Model: Random Forest\n\n"
    )

    f.write(
        "Feature Importance:\n\n"
    )

    f.write(
        feature_summary.to_string(
            index=False
        )
    )

    f.write("\n\nModality Importance:\n\n")

    f.write(
        modality_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nSubject-Level Modality Importance:\n\n"
    )

    f.write(
        subject_modality_pivot.to_string()
    )


# ============================================================
# 17. FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"\n1. {feature_summary_path}")
print(f"2. {modality_summary_path}")
print(f"3. {subject_modality_path}")
print(f"4. {raw_path}")
print(f"5. {summary_path}")

print("\nFeature importance analysis completed successfully.")