import pickle
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 700
WINDOW_SECONDS = 60
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

MAJORITY_THRESHOLD = 0.80

BASE_DIR = Path(__file__).resolve().parents[1]

WESAD_DIR = (
    BASE_DIR
    / "data"
    / "WESAD"
    / "WESAD"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# SUBJECTS
# ============================================================

subjects = sorted(
    [
        folder.name
        for folder in WESAD_DIR.iterdir()
        if folder.is_dir() and folder.name.startswith("S")
    ],
    key=lambda x: int(x[1:])
)

print("========== WESAD SUBJECTS ==========")

print(subjects)

print("\nNumber of subjects:", len(subjects))


# ============================================================
# LABEL DEFINITIONS
# ============================================================

STRESS_LABEL = 2

NON_STRESS_LABELS = [1, 3, 4]

VALID_LABELS = [1, 2, 3, 4]


# ============================================================
# STORAGE
# ============================================================

all_windows = []


# ============================================================
# PROCESS EACH SUBJECT
# ============================================================

for subject in subjects:

    print("\n" + "=" * 60)

    print("Processing:", subject)

    file_path = (
        WESAD_DIR
        / subject
        / f"{subject}.pkl"
    )

    if not file_path.exists():

        print("WARNING: File not found:")
        print(file_path)

        continue

    # --------------------------------------------------------
    # Load subject
    # --------------------------------------------------------

    with open(file_path, "rb") as f:

        data = pickle.load(
            f,
            encoding="latin1"
        )

    labels = data["label"]

    chest = data["signal"]["chest"]

    # --------------------------------------------------------
    # Extract signals
    # --------------------------------------------------------

    ecg = chest["ECG"].flatten()

    eda = chest["EDA"].flatten()

    temperature = chest["Temp"].flatten()

    acc = chest["ACC"]

    # --------------------------------------------------------
    # Number of complete windows
    # --------------------------------------------------------

    num_windows = len(labels) // WINDOW_SIZE

    stress_count = 0

    nonstress_count = 0

    excluded_count = 0

    # --------------------------------------------------------
    # Process windows
    # --------------------------------------------------------

    for window_id in range(num_windows):

        start = window_id * WINDOW_SIZE

        end = start + WINDOW_SIZE

        window_labels = labels[start:end]

        unique, counts = np.unique(
            window_labels,
            return_counts=True
        )

        distribution = dict(
            zip(unique, counts)
        )

        # Count only valid labels

        valid_counts = {
            label: distribution.get(label, 0)
            for label in VALID_LABELS
        }

        dominant_label = max(
            valid_counts,
            key=valid_counts.get
        )

        dominant_count = valid_counts[dominant_label]

        dominance_ratio = (
            dominant_count / WINDOW_SIZE
        )

        # ----------------------------------------------------
        # Assign class
        # ----------------------------------------------------

        if dominance_ratio < MAJORITY_THRESHOLD:

            excluded_count += 1

            continue

        if dominant_label == STRESS_LABEL:

            target = 1

            stress_count += 1

        elif dominant_label in NON_STRESS_LABELS:

            target = 0

            nonstress_count += 1

        else:

            excluded_count += 1

            continue

        # ----------------------------------------------------
        # Calculate simple raw-signal statistics
        #
        # These are NOT our final features yet.
        # They are only used to verify the pipeline.
        # ----------------------------------------------------

        row = {

            "subject": subject,

            "window_id": window_id,

            "label": target,

            "dominant_original_label":
                dominant_label,

            "label_purity":
                dominance_ratio,

            # ECG
            "ecg_mean":
                np.mean(ecg[start:end]),

            "ecg_std":
                np.std(ecg[start:end]),

            # EDA
            "eda_mean":
                np.mean(eda[start:end]),

            "eda_std":
                np.std(eda[start:end]),

            # Temperature
            "temp_mean":
                np.mean(
                    temperature[start:end]
                ),

            "temp_std":
                np.std(
                    temperature[start:end]
                ),

            # Accelerometer magnitude
            "acc_magnitude_mean":
                np.mean(
                    np.sqrt(
                        np.sum(
                            acc[start:end] ** 2,
                            axis=1
                        )
                    )
                ),

            "acc_magnitude_std":
                np.std(
                    np.sqrt(
                        np.sum(
                            acc[start:end] ** 2,
                            axis=1
                        )
                    )
                )
        }

        all_windows.append(row)

    # --------------------------------------------------------
    # Subject summary
    # --------------------------------------------------------

    print(
        f"Complete windows: {num_windows}"
    )

    print(
        f"Stress windows: {stress_count}"
    )

    print(
        f"Non-stress windows: {nonstress_count}"
    )

    print(
        f"Excluded windows: {excluded_count}"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_windows)


# ============================================================
# SAVE
# ============================================================

output_file = (
    RESULTS_DIR
    / "wesad_window_dataset.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("FINAL DATASET SUMMARY")

print("=" * 60)

print(
    "Total usable windows:",
    len(df)
)

print(
    "Subjects represented:",
    df["subject"].nunique()
)

print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print("\nWindows per subject:")

print(
    df.groupby("subject")
      .size()
)

print("\nSaved to:")

print(output_file)