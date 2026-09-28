# ============================================================
# 18_statistical_analysis.py
#
# Statistical analysis of subject-level LOSO F1
#
# 4 Models × 3 pairwise configuration comparisons
# = 12 statistical comparisons
#
# Dataset:
# results/complete_subject_analysis/
# complete_subject_level_results.csv
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    "results/complete_subject_analysis/"
    "complete_subject_level_results.csv"
)

OUTPUT_DIR = "results/statistical_analysis"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "wilcoxon_holm_results_all_models.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "statistical_analysis_report_all_models.txt"
)


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [
    "C1_Heart_HRV",
    "C3_Heart_EDA",
    "C4_Full"
]


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
# PAIRWISE COMPARISONS
# ============================================================

COMPARISONS = [
    ("C1_Heart_HRV", "C3_Heart_EDA"),
    ("C1_Heart_HRV", "C4_Full"),
    ("C3_Heart_EDA", "C4_Full")
]


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(p_values)

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    adjusted_sorted = np.empty(n)

    for i, p in enumerate(sorted_p):

        adjusted_sorted[i] = (
            n - i
        ) * p

    # Enforce monotonicity
    for i in range(1, n):

        adjusted_sorted[i] = max(
            adjusted_sorted[i],
            adjusted_sorted[i - 1]
        )

    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0
    )

    adjusted = np.empty(n)

    adjusted[order] = adjusted_sorted

    return adjusted


# ============================================================
# RANK-BISERIAL EFFECT SIZE
# ============================================================

def rank_biserial_effect(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    differences = x - y

    # Remove zero differences
    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:

        return 0.0

    absolute_differences = np.abs(
        differences
    )

    ranks = pd.Series(
        absolute_differences
    ).rank(
        method="average"
    ).values

    positive_rank_sum = ranks[
        differences > 0
    ].sum()

    negative_rank_sum = ranks[
        differences < 0
    ].sum()

    total_rank_sum = (
        positive_rank_sum
        + negative_rank_sum
    )

    if total_rank_sum == 0:

        return 0.0

    return (
        positive_rank_sum
        - negative_rank_sum
    ) / total_rank_sum


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "STATISTICAL ANALYSIS OF "
        "SUBJECT-LEVEL LOSO F1"
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
    # Load complete subject-level results
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Input: {INPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Subjects: "
        f"{df['test_subject'].nunique()}"
    )

    print(
        f"Models: "
        f"{sorted(df['model'].unique())}"
    )

    print(
        f"Configurations: "
        f"{sorted(df['config'].unique())}"
    )

    # --------------------------------------------------------
    # Validate expected data
    # --------------------------------------------------------

    expected_models = set(
        MODELS
    )

    actual_models = set(
        df["model"].unique()
    )

    if actual_models != expected_models:

        raise ValueError(
            f"Unexpected model set.\n"
            f"Expected: {expected_models}\n"
            f"Found: {actual_models}"
        )

    # --------------------------------------------------------
    # Convert F1 to numeric
    # --------------------------------------------------------

    df["f1"] = pd.to_numeric(
        df["f1"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["f1"]
    )

    # ========================================================
    # COLLECT RESULTS
    # ========================================================

    results = []

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model in MODELS:

        print()
        print("=" * 75)
        print(
            f"MODEL: {model}"
        )
        print("=" * 75)

        df_model = df[
            df["model"] == model
        ].copy()

        # ----------------------------------------------------
        # Subject × configuration F1 table
        # ----------------------------------------------------

        pivot = df_model.pivot(
            index="test_subject",
            columns="config",
            values="f1"
        )

        # Verify required configurations
        missing = [
            config
            for config in CONFIGURATIONS
            if config not in pivot.columns
        ]

        if missing:

            raise ValueError(
                f"Model {model} is missing "
                f"configurations: {missing}"
            )

        pivot = pivot[
            CONFIGURATIONS
        ]

        pivot = pivot.dropna(
            subset=CONFIGURATIONS
        )

        print(
            f"Complete subjects: "
            f"{len(pivot)}"
        )

        # ----------------------------------------------------
        # Pairwise comparisons
        # ----------------------------------------------------

        for config_a, config_b in COMPARISONS:

            x = pivot[
                config_a
            ].values

            y = pivot[
                config_b
            ].values

            differences = x - y

            # ------------------------------------------------
            # Wilcoxon signed-rank test
            # ------------------------------------------------

            if np.allclose(
                differences,
                0
            ):

                statistic = 0.0
                p_value = 1.0

            else:

                test = wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto"
                )

                statistic = float(
                    test.statistic
                )

                p_value = float(
                    test.pvalue
                )

            # ------------------------------------------------
            # Effect size
            # ------------------------------------------------

            effect_size = (
                rank_biserial_effect(
                    x,
                    y
                )
            )

            # ------------------------------------------------
            # Differences
            # ------------------------------------------------

            mean_difference = np.mean(
                differences
            )

            median_difference = np.median(
                differences
            )

            # ------------------------------------------------
            # Subject counts
            # ------------------------------------------------

            a_higher = int(
                np.sum(
                    differences > 0
                )
            )

            b_higher = int(
                np.sum(
                    differences < 0
                )
            )

            equal = int(
                np.sum(
                    differences == 0
                )
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            results.append({

                "model":
                    model,

                "comparison":
                    f"{config_a} vs {config_b}",

                "config_a":
                    config_a,

                "config_b":
                    config_b,

                "n_subjects":
                    len(x),

                "mean_f1_difference":
                    mean_difference,

                "median_f1_difference":
                    median_difference,

                "wilcoxon_statistic":
                    statistic,

                "raw_p_value":
                    p_value,

                "rank_biserial_effect":
                    effect_size,

                "a_higher_subjects":
                    a_higher,

                "b_higher_subjects":
                    b_higher,

                "equal_subjects":
                    equal
            })

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print()
            print(
                f"{config_a}  vs  {config_b}"
            )

            print(
                f"  Mean F1 difference: "
                f"{mean_difference:.4f}"
            )

            print(
                f"  Median F1 difference: "
                f"{median_difference:.4f}"
            )

            print(
                f"  Wilcoxon statistic: "
                f"{statistic:.4f}"
            )

            print(
                f"  Raw p-value: "
                f"{p_value:.6f}"
            )

            print(
                f"  Rank-biserial effect: "
                f"{effect_size:.4f}"
            )

            print(
                f"  {config_a} higher: "
                f"{a_higher}"
            )

            print(
                f"  {config_b} higher: "
                f"{b_higher}"
            )

            print(
                f"  Equal: "
                f"{equal}"
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # HOLM CORRECTION
    # ========================================================

    results_df[
        "holm_adjusted_p"
    ] = holm_correction(
        results_df[
            "raw_p_value"
        ].values
    )

    # --------------------------------------------------------
    # Significance
    # --------------------------------------------------------

    results_df[
        "significant_after_holm"
    ] = (
        results_df[
            "holm_adjusted_p"
        ] < 0.05
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 75)
    print(
        "FINAL STATISTICAL RESULTS"
    )
    print("=" * 75)

    display_columns = [

        "model",

        "comparison",

        "n_subjects",

        "mean_f1_difference",

        "median_f1_difference",

        "wilcoxon_statistic",

        "raw_p_value",

        "holm_adjusted_p",

        "rank_biserial_effect",

        "a_higher_subjects",

        "b_higher_subjects",

        "equal_subjects",

        "significant_after_holm"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "STATISTICAL ANALYSIS OF "
        "SUBJECT-LEVEL LOSO F1"
    )

    report.append(
        "=" * 65
    )

    report.append(
        f"Number of subjects: "
        f"{df['test_subject'].nunique()}"
    )

    report.append(
        "Models: LR, SVM, RF, XGB"
    )

    report.append(
        "Configurations: "
        "C1_Heart_HRV, C3_Heart_EDA, C4_Full"
    )

    report.append("")

    report.append(
        "Statistical test:"
    )

    report.append(
        "Wilcoxon signed-rank test "
        "on paired subject-level F1 scores."
    )

    report.append("")

    report.append(
        "Multiple-comparison correction:"
    )

    report.append(
        "Holm-Bonferroni correction across "
        "all 12 model/configuration comparisons."
    )

    report.append("")

    report.append(
        "Effect size:"
    )

    report.append(
        "Rank-biserial correlation."
    )

    report.append("")

    # --------------------------------------------------------
    # Detailed results
    # --------------------------------------------------------

    for _, row in results_df.iterrows():

        report.append(
            "-" * 65
        )

        report.append(
            f"Model: {row['model']}"
        )

        report.append(
            f"Comparison: "
            f"{row['comparison']}"
        )

        report.append(
            f"Mean F1 difference: "
            f"{row['mean_f1_difference']:.4f}"
        )

        report.append(
            f"Median F1 difference: "
            f"{row['median_f1_difference']:.4f}"
        )

        report.append(
            f"Wilcoxon statistic: "
            f"{row['wilcoxon_statistic']:.4f}"
        )

        report.append(
            f"Raw p-value: "
            f"{row['raw_p_value']:.6f}"
        )

        report.append(
            f"Holm-adjusted p-value: "
            f"{row['holm_adjusted_p']:.6f}"
        )

        report.append(
            f"Rank-biserial effect: "
            f"{row['rank_biserial_effect']:.4f}"
        )

        report.append(
            f"Configuration A higher: "
            f"{int(row['a_higher_subjects'])}"
        )

        report.append(
            f"Configuration B higher: "
            f"{int(row['b_higher_subjects'])}"
        )

        report.append(
            f"Equal: "
            f"{int(row['equal_subjects'])}"
        )

        report.append(
            f"Significant after Holm: "
            f"{row['significant_after_holm']}"
        )

        report.append("")

    # ========================================================
    # SUMMARY
    # ========================================================

    significant_count = int(
        results_df[
            "significant_after_holm"
        ].sum()
    )

    report.append(
        "=" * 65
    )

    report.append(
        f"Comparisons significant after "
        f"Holm correction: "
        f"{significant_count}/12"
    )

    report.append("")

    report.append(
        "Interpretation should consider "
        "the paired subject-level design, "
        "the limited number of subjects (n=15), "
        "and substantial inter-subject variability."
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(
        f"CSV: {RESULTS_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print()
    print(
        "Statistical analysis completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()