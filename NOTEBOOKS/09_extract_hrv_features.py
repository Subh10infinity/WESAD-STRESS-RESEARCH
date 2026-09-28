import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# SETTINGS
# ============================================================

FS = 700
WINDOW_SECONDS = 60
WINDOW_SIZE = FS * WINDOW_SECONDS

LOWCUT = 5
HIGHCUT = 20
FILTER_ORDER = 3

MIN_PEAK_DISTANCE = int(0.30 * FS)

RR_MIN = 0.30
RR_MAX = 1.50

VALID_LABELS = [1, 2, 3, 4]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "WESAD" / "WESAD"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


subjects = [
    "S2", "S3", "S4", "S5", "S6",
    "S7", "S8", "S9", "S10", "S11",
    "S13", "S14", "S15", "S16", "S17"
]


# ============================================================
# ECG FILTER
# ============================================================

def bandpass_filter(signal):

    nyquist = 0.5 * FS

    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    b, a = butter(
        FILTER_ORDER,
        [low, high],
        btype="band"
    )

    return filtfilt(b, a, signal)


# ============================================================
# ECG R-PEAK DETECTION
# ============================================================

def detect_r_peaks(ecg):

    filtered_ecg = bandpass_filter(ecg)

    threshold = (
        np.mean(filtered_ecg)
        + 0.5 * np.std(filtered_ecg)
    )

    peaks, _ = find_peaks(
        filtered_ecg,
        height=threshold,
        distance=MIN_PEAK_DISTANCE,
        prominence=0.5 * np.std(filtered_ecg)
    )

    return peaks


# ============================================================
# HRV FEATURES
# ============================================================

def extract_hrv_features(ecg):

    peaks = detect_r_peaks(ecg)

    # Need enough peaks to calculate RR intervals
    if len(peaks) < 3:
        return {
            "mean_hr": np.nan,
            "mean_rr": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "num_peaks": len(peaks),
            "invalid_rr_percentage": np.nan
        }

    # Convert peak differences to seconds
    rr = np.diff(peaks) / FS

    # Remove implausible RR intervals
    valid_rr = rr[
        (rr >= RR_MIN) &
        (rr <= RR_MAX)
    ]

    if len(valid_rr) < 2:

        return {
            "mean_hr": np.nan,
            "mean_rr": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "num_peaks": len(peaks),
            "invalid_rr_percentage": 100.0
        }

    # --------------------------------------------------------
    # Mean RR
    # --------------------------------------------------------

    mean_rr = np.mean(valid_rr)

    # --------------------------------------------------------
    # Mean HR
    # --------------------------------------------------------

    mean_hr = 60 / mean_rr

    # --------------------------------------------------------
    # SDNN
    # --------------------------------------------------------

    sdnn = np.std(
        valid_rr,
        ddof=1
    )

    # --------------------------------------------------------
    # RMSSD
    # --------------------------------------------------------

    rr_diff = np.diff(valid_rr)

    rmssd = np.sqrt(
        np.mean(rr_diff ** 2)
    )

    # --------------------------------------------------------
    # Invalid RR percentage
    # --------------------------------------------------------

    invalid_percentage = (
        1 -
        len(valid_rr) / len(rr)
    ) * 100

    return {
        "mean_hr": mean_hr,
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "num_peaks": len(peaks),
        "invalid_rr_percentage": invalid_percentage
    }


# ============================================================
# PROCESS SUBJECT
# ============================================================

def process_subject(subject):

    file_path = (
        DATA_DIR /
        subject /
        f"{subject}.pkl"
    )

    print(f"Processing {subject}...")

    with open(file_path, "rb") as f:

        data = pickle.load(
            f,
            encoding="latin1"
        )

    chest = data["signal"]["chest"]

    ecg = chest["ECG"].flatten()
    eda = chest["EDA"].flatten()
    temp = chest["Temp"].flatten()
    acc = chest["ACC"]

    labels = np.asarray(
        data["label"]
    )

    total_windows = (
        len(ecg) // WINDOW_SIZE
    )

    subject_results = []

    for window_id in range(total_windows):

        start = (
            window_id *
            WINDOW_SIZE
        )

        end = start + WINDOW_SIZE

        window_labels = labels[start:end]

        if len(window_labels) < WINDOW_SIZE:
            continue

        # ----------------------------------------------------
        # Only use labels relevant to our binary experiment
        # ----------------------------------------------------

        valid_labels = window_labels[
            np.isin(
                window_labels,
                VALID_LABELS
            )
        ]

        if len(valid_labels) == 0:
            continue

        # Find dominant valid label
        values, counts = np.unique(
            valid_labels,
            return_counts=True
        )

        dominant_label = values[
            np.argmax(counts)
        ]

        dominant_count = np.max(counts)

        purity = (
            dominant_count /
            WINDOW_SIZE
        )

        # Require at least 80% purity
        if purity < 0.80:
            continue

        # ----------------------------------------------------
        # Convert original WESAD label to binary target
        #
        # 2 = Stress
        # 1,3,4 = Non-stress
        # ----------------------------------------------------

        if dominant_label == 2:

            target = 1

        elif dominant_label in [1, 3, 4]:

            target = 0

        else:

            continue

        # ----------------------------------------------------
        # Extract signals
        # ----------------------------------------------------

        ecg_window = ecg[start:end]

        eda_window = eda[start:end]

        temp_window = temp[start:end]

        acc_window = acc[start:end]

        # ----------------------------------------------------
        # ECG / HRV
        # ----------------------------------------------------

        hrv = extract_hrv_features(
            ecg_window
        )

        # ----------------------------------------------------
        # EDA
        # ----------------------------------------------------

        eda_mean = np.mean(
            eda_window
        )

        eda_std = np.std(
            eda_window
        )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        temp_mean = np.mean(
            temp_window
        )

        temp_std = np.std(
            temp_window
        )

        # ----------------------------------------------------
        # Accelerometer magnitude
        # ----------------------------------------------------

        acc_magnitude = np.sqrt(
            np.sum(
                acc_window ** 2,
                axis=1
            )
        )

        acc_mean = np.mean(
            acc_magnitude
        )

        acc_std = np.std(
            acc_magnitude
        )

        # ----------------------------------------------------
        # Store row
        # ----------------------------------------------------

        row = {

            "subject": subject,

            "window_id": window_id,

            "label": target,

            "dominant_original_label":
                dominant_label,

            "label_purity":
                purity,

            # ECG / HRV
            "mean_hr":
                hrv["mean_hr"],

            "mean_rr":
                hrv["mean_rr"],

            "sdnn":
                hrv["sdnn"],

            "rmssd":
                hrv["rmssd"],

            "num_peaks":
                hrv["num_peaks"],

            "invalid_rr_percentage":
                hrv["invalid_rr_percentage"],

            # EDA
            "eda_mean":
                eda_mean,

            "eda_std":
                eda_std,

            # Temperature
            "temp_mean":
                temp_mean,

            "temp_std":
                temp_std,

            # Accelerometer
            "acc_mean":
                acc_mean,

            "acc_std":
                acc_std
        }

        subject_results.append(row)

    return subject_results


# ============================================================
# PROCESS ALL SUBJECTS
# ============================================================

all_results = []

for subject in subjects:

    results = process_subject(subject)

    all_results.extend(results)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_results
)


# ============================================================
# REMOVE WINDOWS WHERE HRV FAILED
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "mean_hr",
        "mean_rr",
        "sdnn",
        "rmssd"
    ]
).reset_index(drop=True)

after = len(df)


# ============================================================
# SAVE
# ============================================================

output_path = (
    RESULTS_DIR /
    "wesad_hrv_feature_dataset.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("HRV FEATURE EXTRACTION COMPLETE")
print("=" * 80)

print(
    f"Windows before HRV filtering : {before}"
)

print(
    f"Windows after HRV filtering  : {after}"
)

print(
    f"Removed windows              : {before - after}"
)

print(
    f"Subjects                     : "
    f"{df['subject'].nunique()}"
)

print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print("\nFeature columns:")

for column in df.columns:
    print(" -", column)

print("\nSaved to:")

print(output_path)