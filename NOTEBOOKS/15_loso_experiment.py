from pathlib import Path

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
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "wesad_hrv_feature_dataset.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOSO EXPERIMENT")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows    : {len(df)}")
print(f"Subjects: {df['subject'].nunique()}")


# ============================================================
# FEATURE CONFIGURATIONS
# ============================================================

CONFIGURATIONS = {

    "C1_Heart_HRV": [
        "mean_hr",
        "sdnn",
        "rmssd",
    ],

    "C2_EDA": [
        "eda_mean",
        "eda_std",
    ],

    "C3_Heart_EDA": [
        "mean_hr",
        "sdnn",
        "rmssd",
        "eda_mean",
        "eda_std",
    ],

    "C4_Full_Multimodal": [
        "mean_hr",
        "sdnn",
        "rmssd",
        "eda_mean",
        "eda_std",
        "temp_mean",
        "temp_std",
        "acc_mean",
        "acc_std",
    ],
}


# ============================================================
# MODEL CREATION
# ============================================================

def create_models(y_train):

    class_counts = np.bincount(y_train)

    negative_count = class_counts[0]
    positive_count = class_counts[1]

    scale_pos_weight = (
        negative_count / positive_count
    )

    models = {

        "Logistic_Regression":
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),

        "SVM":
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),

        "Random_Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
    }

    return models


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    return {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                y_true,
                y_prob
            ),
    }


# ============================================================
# SUBJECTS
# ============================================================

subjects = sorted(
    df["subject"].unique()
)

print("\nSubjects:")
print(subjects)


# ============================================================
# RESULT STORAGE
# ============================================================

fold_results = []


# ============================================================
# MAIN LOSO EXPERIMENT
# ============================================================

for config_name, features in CONFIGURATIONS.items():

    print("\n")
    print("=" * 80)
    print(f"CONFIGURATION: {config_name}")
    print("=" * 80)

    print("Features:")
    print(features)

    X = df[features].copy()
    y = df["label"].astype(int)
    subject_ids = df["subject"]


    # ========================================================
    # LOSO
    # ========================================================

    for test_subject in subjects:

        print(
            f"\nTesting subject: {test_subject}"
        )

        train_mask = (
            subject_ids != test_subject
        )

        test_mask = (
            subject_ids == test_subject
        )

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]


        # ====================================================
        # SCALING FOR LR + SVM
        # ====================================================

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )


        # ====================================================
        # MODELS
        # ====================================================

        models = create_models(
            y_train.to_numpy()
        )


        for model_name, model in models.items():

            print(
                f"  Running {model_name}..."
            )


            # =================================================
            # SELECT DATA REPRESENTATION
            # =================================================

            if model_name in [
                "Logistic_Regression",
                "SVM"
            ]:

                X_train_model = X_train_scaled
                X_test_model = X_test_scaled

            else:

                X_train_model = X_train
                X_test_model = X_test


            # =================================================
            # TRAIN
            # =================================================

            model.fit(
                X_train_model,
                y_train
            )


            # =================================================
            # PREDICTION
            # =================================================

            y_pred = model.predict(
                X_test_model
            )

            y_prob = model.predict_proba(
                X_test_model
            )[:, 1]


            # =================================================
            # METRICS
            # =================================================

            metrics = calculate_metrics(
                y_test,
                y_pred,
                y_prob
            )


            fold_results.append({

                "configuration":
                    config_name,

                "model":
                    model_name,

                "test_subject":
                    test_subject,

                "n_train":
                    len(X_train),

                "n_test":
                    len(X_test),

                "accuracy":
                    metrics["accuracy"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1":
                    metrics["f1"],

                "auc":
                    metrics["auc"],
            })


# ============================================================
# FOLD RESULTS
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)


fold_path = (
    RESULTS_DIR
    / "loso_fold_results.csv"
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = (
    fold_df
    .groupby(
        [
            "configuration",
            "model"
        ]
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
        [
            "mean",
            "std"
        ]
    )
)


summary_path = (
    RESULTS_DIR
    / "loso_summary.csv"
)

summary_df.to_csv(
    summary_path
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 100)
print("LOSO EXPERIMENT COMPLETE")
print("=" * 100)

print("\nMean ± Standard Deviation Across Subjects:\n")


for configuration in CONFIGURATIONS:

    print("\n" + "-" * 80)
    print(configuration)
    print("-" * 80)

    config_results = (
        fold_df[
            fold_df["configuration"]
            == configuration
        ]
    )

    for model_name in [
        "Logistic_Regression",
        "SVM",
        "Random_Forest",
        "XGBoost"
    ]:

        model_results = (
            config_results[
                config_results["model"]
                == model_name
            ]
        )

        print(
            f"\n{model_name}"
        )

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc"
        ]:

            mean_value = (
                model_results[metric]
                .mean()
            )

            std_value = (
                model_results[metric]
                .std()
            )

            print(
                f"  {metric:<10}: "
                f"{mean_value:.4f} ± "
                f"{std_value:.4f}"
            )


# ============================================================
# FILE LOCATIONS
# ============================================================

print("\n")
print("=" * 80)
print("RESULT FILES")
print("=" * 80)

print("\nFold-level results:")
print(fold_path)

print("\nSummary results:")
print(summary_path)