# ============================================================
# 19_complete_subject_analysis.py
#
# Complete subject-level LOSO analysis
#
# 4 Models × 4 Configurations × 15 Subjects = 240 evaluations
#
# Dataset:
#   results/wesad_clean_feature_dataset.csv
#
# Output:
#   results/complete_subject_analysis/
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "results/wesad_clean_feature_dataset.csv"

OUTPUT_DIR = "results/complete_subject_analysis"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "complete_subject_level_results.csv"
)


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGURATIONS = {

    "C1_Heart_HRV": [
        "mean_hr",
        "sdnn",
        "rmssd"
    ],

    "C2_EDA": [
        "eda_mean",
        "eda_std"
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
# MODELS
# ============================================================

MODELS = [
    "LR",
    "SVM",
    "RF",
    "XGB"
]


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(model_name, y_train):

    if model_name == "LR":

        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )

    elif model_name == "SVM":

        return SVC(
            class_weight="balanced",
            probability=True,
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

        negative = np.sum(
            y_train == 0
        )

        positive = np.sum(
            y_train == 1
        )

        scale_pos_weight = (
            negative / positive
            if positive > 0
            else 1.0
        )

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


# ============================================================
# SAFE AUC
# ============================================================

def calculate_auc(y_true, y_prob):

    try:

        return roc_auc_score(
            y_true,
            y_prob
        )

    except ValueError:

        return np.nan


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "COMPLETE SUBJECT-LEVEL LOSO ANALYSIS"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load clean dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Dataset: {INPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Subjects: "
        f"{df['subject'].nunique()}"
    )

    print(
        f"Class distribution:"
    )

    print(
        df["label"].value_counts().sort_index()
    )

    subjects = sorted(
        df["subject"].unique()
    )

    print()
    print(
        f"Subjects: {subjects}"
    )

    # ========================================================
    # VALIDATE DATA
    # ========================================================

    required_columns = set(
        ["subject", "label"]
    )

    for features in CONFIGURATIONS.values():

        required_columns.update(
            features
        )

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    total_experiments = (
        len(CONFIGURATIONS)
        * len(MODELS)
        * len(subjects)
    )

    experiment_number = 0

    # ========================================================
    # LOOP CONFIGURATIONS
    # ========================================================

    for config_name, features in CONFIGURATIONS.items():

        print()
        print("=" * 75)
        print(
            f"CONFIGURATION: {config_name}"
        )
        print(
            f"Features: {features}"
        )
        print("=" * 75)

        # ----------------------------------------------------
        # LOOP MODELS
        # ----------------------------------------------------

        for model_name in MODELS:

            print()
            print(
                f"  MODEL: {model_name}"
            )

            # ------------------------------------------------
            # LOOP HELD-OUT SUBJECTS
            # ------------------------------------------------

            for test_subject in subjects:

                experiment_number += 1

                print(
                    f"    [{experiment_number}/"
                    f"{total_experiments}] "
                    f"Testing {test_subject}"
                )

                # --------------------------------------------
                # Train / test split by subject
                # --------------------------------------------

                train_df = df[
                    df["subject"] != test_subject
                ].copy()

                test_df = df[
                    df["subject"] == test_subject
                ].copy()

                X_train = train_df[
                    features
                ].values

                y_train = train_df[
                    "label"
                ].values

                X_test = test_df[
                    features
                ].values

                y_test = test_df[
                    "label"
                ].values

                # --------------------------------------------
                # Scale ONLY for LR and SVM
                #
                # Scaler is fitted only on training subjects.
                # --------------------------------------------

                if model_name in [
                    "LR",
                    "SVM"
                ]:

                    scaler = StandardScaler()

                    X_train_model = (
                        scaler.fit_transform(
                            X_train
                        )
                    )

                    X_test_model = (
                        scaler.transform(
                            X_test
                        )
                    )

                else:

                    X_train_model = X_train

                    X_test_model = X_test

                # --------------------------------------------
                # Create model
                # --------------------------------------------

                model = create_model(
                    model_name,
                    y_train
                )

                # --------------------------------------------
                # Train
                # --------------------------------------------

                model.fit(
                    X_train_model,
                    y_train
                )

                # --------------------------------------------
                # Predictions
                # --------------------------------------------

                y_pred = model.predict(
                    X_test_model
                )

                # --------------------------------------------
                # Probability / decision score
                # --------------------------------------------

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    y_prob = model.predict_proba(
                        X_test_model
                    )[:, 1]

                elif hasattr(
                    model,
                    "decision_function"
                ):

                    y_prob = model.decision_function(
                        X_test_model
                    )

                else:

                    y_prob = y_pred.astype(
                        float
                    )

                # --------------------------------------------
                # Confusion matrix components
                # --------------------------------------------

                tn = int(
                    np.sum(
                        (y_test == 0)
                        & (y_pred == 0)
                    )
                )

                fp = int(
                    np.sum(
                        (y_test == 0)
                        & (y_pred == 1)
                    )
                )

                fn = int(
                    np.sum(
                        (y_test == 1)
                        & (y_pred == 0)
                    )
                )

                tp = int(
                    np.sum(
                        (y_test == 1)
                        & (y_pred == 1)
                    )
                )

                # --------------------------------------------
                # Metrics
                # --------------------------------------------

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

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

                auc = calculate_auc(
                    y_test,
                    y_prob
                )

                # --------------------------------------------
                # Save result
                # --------------------------------------------

                results.append({

                    "config":
                        config_name,

                    "model":
                        model_name,

                    "test_subject":
                        test_subject,

                    "n_test_windows":
                        len(test_df),

                    "actual_nonstress":
                        int(
                            np.sum(
                                y_test == 0
                            )
                        ),

                    "actual_stress":
                        int(
                            np.sum(
                                y_test == 1
                            )
                        ),

                    "predicted_nonstress":
                        int(
                            np.sum(
                                y_pred == 0
                            )
                        ),

                    "predicted_stress":
                        int(
                            np.sum(
                                y_pred == 1
                            )
                        ),

                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                    "TP": tp,

                    "accuracy":
                        accuracy,

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "f1":
                        f1,

                    "auc":
                        auc
                })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SORT
    # ========================================================

    results_df = results_df.sort_values(
        by=[
            "config",
            "model",
            "test_subject"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 75)
    print("VALIDATION")
    print("=" * 75)

    print(
        f"Total rows: {len(results_df)}"
    )

    expected_rows = (
        len(CONFIGURATIONS)
        * len(MODELS)
        * len(subjects)
    )

    print(
        f"Expected rows: {expected_rows}"
    )

    print()

    print(
        "Rows per configuration:"
    )

    print(
        results_df[
            "config"
        ].value_counts()
    )

    print()

    print(
        "Rows per model:"
    )

    print(
        results_df[
            "model"
        ].value_counts()
    )

    print()

    print(
        "Rows per configuration/model:"
    )

    print(
        results_df.groupby(
            ["config", "model"]
        ).size()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("MEAN SUBJECT-LEVEL RESULTS")
    print("=" * 75)

    summary = (
        results_df
        .groupby(
            ["config", "model"]
        )
        [
            [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc"
            ]
        ]
        .agg(
            ["mean", "std"]
        )
        .round(4)
    )

    print(
        summary.to_string()
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("=" * 75)
    print("COMPLETE")
    print("=" * 75)

    print(
        f"Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "The file now contains:"
    )

    print(
        "4 configurations × "
        "4 models × "
        "15 subjects = "
        f"{len(results_df)} rows"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()