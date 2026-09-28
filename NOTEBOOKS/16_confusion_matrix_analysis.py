import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "results/wesad_clean_feature_dataset.csv"

OUTPUT_DIR = "results/confusion_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD CLEAN DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("CONFUSION MATRIX + SUBJECT-LEVEL ERROR ANALYSIS")
print("=" * 70)

print(f"\nDataset: {DATA_PATH}")
print(f"Rows: {len(df)}")
print(f"Subjects: {df['subject'].nunique()}")

print("\nClass distribution:")
print(df["label"].value_counts().sort_index())


# ============================================================
# 3. SENSOR CONFIGURATIONS
# ============================================================

CONFIGS = {

    "C1_Heart_HRV": [
        "mean_hr",
        "sdnn",
        "rmssd"
    ],

    "C3_Heart_EDA": [
        "mean_hr",
        "sdnn",
        "rmssd",
        "eda_mean",
        "eda_std"
    ],

    "C4_Full": [
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
}


# ============================================================
# 4. MODELS
# ============================================================

def create_model(model_name, y_train):

    if model_name == "LR":

        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )

    elif model_name == "SVM":

        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        )

    elif model_name == "RF":

        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "XGB":

        negative = np.sum(y_train == 0)
        positive = np.sum(y_train == 1)

        scale_pos_weight = negative / positive

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# 5. SELECT KEY EXPERIMENTS
# ============================================================

# We focus on Random Forest because:
# - It performed strongly in the primary LOSO experiment.
# - It gives a useful comparison across sensor configurations.
#
# C4 XGBoost is also included because it had the highest mean AUC
# in the primary experiment.

EXPERIMENTS = [
    ("C1_Heart_HRV", "RF"),
    ("C3_Heart_EDA", "RF"),
    ("C4_Full", "RF"),
    ("C4_Full", "XGB")
]


# ============================================================
# 6. STORAGE
# ============================================================

fold_results = []
all_predictions = []


# ============================================================
# 7. LOSO EXPERIMENT
# ============================================================

subjects = sorted(df["subject"].unique())

print("\nSubjects:")
print(subjects)

print("\nStarting LOSO analysis...\n")


for config_name, model_name in EXPERIMENTS:

    features = CONFIGS[config_name]

    print("-" * 70)
    print(f"Configuration: {config_name}")
    print(f"Model: {model_name}")
    print(f"Features: {features}")
    print("-" * 70)

    for test_subject in subjects:

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        train_df = df[df["subject"] != test_subject].copy()
        test_df = df[df["subject"] == test_subject].copy()

        X_train = train_df[features].values
        y_train = train_df["label"].values

        X_test = test_df[features].values
        y_test = test_df["label"].values

        # ----------------------------------------------------
        # SCALE ONLY FOR LR / SVM
        # ----------------------------------------------------

        if model_name in ["LR", "SVM"]:

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = create_model(model_name, y_train)

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):

            y_prob = model.predict_proba(X_test)[:, 1]

        else:

            y_prob = None

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        ).ravel()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        if y_prob is not None and len(np.unique(y_test)) == 2:

            auc = roc_auc_score(
                y_test,
                y_prob
            )

        else:

            auc = np.nan

        # ----------------------------------------------------
        # SAVE FOLD RESULT
        # ----------------------------------------------------

        fold_results.append({

            "config": config_name,
            "model": model_name,
            "test_subject": test_subject,

            "n_test_windows": len(test_df),

            "actual_nonstress": int(np.sum(y_test == 0)),
            "actual_stress": int(np.sum(y_test == 1)),

            "predicted_nonstress": int(np.sum(y_pred == 0)),
            "predicted_stress": int(np.sum(y_pred == 1)),

            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),

            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        })

        # ----------------------------------------------------
        # SAVE INDIVIDUAL PREDICTIONS
        # ----------------------------------------------------

        for i in range(len(test_df)):

            all_predictions.append({

                "config": config_name,
                "model": model_name,

                "test_subject": test_subject,

                "window_id": test_df.iloc[i]["window_id"]
                if "window_index" in test_df.columns
                else i,

                "true_label": int(y_test[i]),
                "predicted_label": int(y_pred[i]),

                "correct": int(y_test[i] == y_pred[i]),

                "probability_stress":
                    float(y_prob[i])
                    if y_prob is not None
                    else np.nan
            })


# ============================================================
# 8. DATAFRAMES
# ============================================================

fold_df = pd.DataFrame(fold_results)

prediction_df = pd.DataFrame(all_predictions)


# ============================================================
# 9. SAVE SUBJECT-LEVEL RESULTS
# ============================================================

fold_path = os.path.join(
    OUTPUT_DIR,
    "subject_level_confusion_results.csv"
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# 10. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "individual_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 11. AGGREGATE CONFUSION MATRICES
# ============================================================

aggregate_results = []

for config_name, model_name in EXPERIMENTS:

    subset = fold_df[
        (fold_df["config"] == config_name) &
        (fold_df["model"] == model_name)
    ]

    total_TN = subset["TN"].sum()
    total_FP = subset["FP"].sum()
    total_FN = subset["FN"].sum()
    total_TP = subset["TP"].sum()

    total = (
        total_TN +
        total_FP +
        total_FN +
        total_TP
    )

    accuracy = (
        total_TN + total_TP
    ) / total

    precision = (
        total_TP /
        (total_TP + total_FP)
        if (total_TP + total_FP) > 0
        else 0
    )

    recall = (
        total_TP /
        (total_TP + total_FN)
        if (total_TP + total_FN) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    aggregate_results.append({

        "config": config_name,
        "model": model_name,

        "TN": int(total_TN),
        "FP": int(total_FP),
        "FN": int(total_FN),
        "TP": int(total_TP),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


aggregate_df = pd.DataFrame(
    aggregate_results
)


aggregate_path = os.path.join(
    OUTPUT_DIR,
    "aggregate_confusion_matrices.csv"
)

aggregate_df.to_csv(
    aggregate_path,
    index=False
)


# ============================================================
# 12. SUBJECT-LEVEL COMPARISON
# ============================================================

rf_comparison = fold_df[
    fold_df["model"] == "RF"
].copy()

pivot_f1 = rf_comparison.pivot(
    index="test_subject",
    columns="config",
    values="f1"
)

pivot_recall = rf_comparison.pivot(
    index="test_subject",
    columns="config",
    values="recall"
)

pivot_accuracy = rf_comparison.pivot(
    index="test_subject",
    columns="config",
    values="accuracy"
)


pivot_f1_path = os.path.join(
    OUTPUT_DIR,
    "subject_f1_comparison_RF.csv"
)

pivot_recall_path = os.path.join(
    OUTPUT_DIR,
    "subject_recall_comparison_RF.csv"
)

pivot_accuracy_path = os.path.join(
    OUTPUT_DIR,
    "subject_accuracy_comparison_RF.csv"
)


pivot_f1.to_csv(pivot_f1_path)
pivot_recall.to_csv(pivot_recall_path)
pivot_accuracy.to_csv(pivot_accuracy_path)


# ============================================================
# 13. C4 VS C1 SUBJECT IMPROVEMENT
# ============================================================

if (
    "C1_Heart_HRV" in pivot_f1.columns and
    "C4_Full" in pivot_f1.columns
):

    comparison = pd.DataFrame({

        "C1_F1": pivot_f1["C1_Heart_HRV"],

        "C4_F1": pivot_f1["C4_Full"]

    })

    comparison["C4_minus_C1_F1"] = (
        comparison["C4_F1"] -
        comparison["C1_F1"]
    )

    comparison["C4_better"] = (
        comparison["C4_minus_C1_F1"] > 0
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "C4_vs_C1_subject_comparison.csv"
    )

    comparison.to_csv(
        comparison_path
    )


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("AGGREGATE CONFUSION MATRICES")
print("=" * 70)

print(
    aggregate_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n")
print("=" * 70)
print("RANDOM FOREST SUBJECT-LEVEL F1")
print("=" * 70)

print(
    pivot_f1.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


print("\n")
print("=" * 70)
print("RANDOM FOREST SUBJECT-LEVEL RECALL")
print("=" * 70)

print(
    pivot_recall.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"\n1. {fold_path}")
print(f"2. {prediction_path}")
print(f"3. {aggregate_path}")
print(f"4. {pivot_f1_path}")
print(f"5. {pivot_recall_path}")
print(f"6. {pivot_accuracy_path}")

if os.path.exists(
    os.path.join(
        OUTPUT_DIR,
        "C4_vs_C1_subject_comparison.csv"
    )
):
    print(
        "7. "
        + os.path.join(
            OUTPUT_DIR,
            "C4_vs_C1_subject_comparison.csv"
        )
    )

print("\nAnalysis completed successfully.")