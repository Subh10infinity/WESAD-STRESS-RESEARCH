import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# SETTINGS
# ============================================================

FS = 700                    # WESAD chest ECG sampling rate
WINDOW_SECONDS = 60
WINDOW_SIZE = FS * WINDOW_SECONDS

LOWCUT = 5
HIGHCUT = 20
FILTER_ORDER = 3

MIN_PEAK_DISTANCE = int(0.30 * FS)   # max ~200 BPM
RR_MIN = 0.30                         # 200 BPM
RR_MAX = 1.50                         # 40 BPM


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "WESAD" / "WESAD"


# WESAD subjects available in the dataset
subjects = [
    "S2", "S3", "S4", "S5", "S6",
    "S7", "S8", "S9", "S10", "S11",
    "S13", "S14", "S15", "S16", "S17"
]


# ============================================================
# ECG BANDPASS FILTER
# ============================================================

def bandpass_filter(signal, fs, lowcut, highcut, order=3):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    filtered = filtfilt(b, a, signal)

    return filtered


# ============================================================
# FIND R-PEAKS
# ============================================================

def detect_r_peaks(ecg):

    filtered_ecg = bandpass_filter(
        ecg,
        FS,
        LOWCUT,
        HIGHCUT,
        FILTER_ORDER
    )

    threshold = np.mean(filtered_ecg) + 0.5 * np.std(filtered_ecg)

    peaks, properties = find_peaks(
        filtered_ecg,
        height=threshold,
        distance=MIN_PEAK_DISTANCE,
        prominence=0.5 * np.std(filtered_ecg)
    )

    return peaks, filtered_ecg


# ============================================================
# CALCULATE HRV FEATURES
# ============================================================

def calculate_features(peaks):

    if len(peaks) < 3:
        return None

    rr = np.diff(peaks) / FS

    # Keep only physiologically plausible RR intervals
    valid_rr = rr[(rr >= RR_MIN) & (rr <= RR_MAX)]

    if len(valid_rr) < 2:
        return None

    mean_rr = np.mean(valid_rr)

    mean_hr = 60 / mean_rr

    min_hr = 60 / np.max(valid_rr)

    max_hr = 60 / np.min(valid_rr)

    # SDNN
    sdnn = np.std(valid_rr, ddof=1)

    # RMSSD
    successive_differences = np.diff(valid_rr)

    rmssd = np.sqrt(
        np.mean(successive_differences ** 2)
    )

    invalid_percentage = (
        1 - len(valid_rr) / len(rr)
    ) * 100

    return {
        "num_peaks": len(peaks),
        "mean_rr": mean_rr,
        "mean_hr": mean_hr,
        "min_hr": min_hr,
        "max_hr": max_hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "invalid_rr_percentage": invalid_percentage
    }


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(subject):

    file_path = DATA_DIR / subject / f"{subject}.pkl"

    print(f"\nProcessing {subject}...")

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ecg = data["signal"]["chest"]["ECG"].flatten()
    labels = np.asarray(data["label"])

    total_windows = len(ecg) // WINDOW_SIZE

    for window_id in range(total_windows):

        start = window_id * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window_labels = labels[start:end]

        # We want a PURE stress window for validation
        if len(window_labels) < WINDOW_SIZE:
            continue

        if np.all(window_labels == 2):

            ecg_window = ecg[start:end]

            peaks, filtered = detect_r_peaks(ecg_window)

            features = calculate_features(peaks)

            if features is None:
                print(f"{subject}: Window {window_id} -> insufficient RR data")
                return None

            return {
                "subject": subject,
                "window_id": window_id,
                **features
            }

    print(f"{subject}: No pure stress window found.")

    return None


# ============================================================
# RUN VALIDATION
# ============================================================

results = []

for subject in subjects:

    result = process_subject(subject)

    if result is not None:
        results.append(result)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("ECG R-PEAK VALIDATION RESULTS")
print("=" * 80)

print(
    f"{'Subject':<10}"
    f"{'Window':<10}"
    f"{'Peaks':<10}"
    f"{'Mean HR':<12}"
    f"{'Min HR':<12}"
    f"{'Max HR':<12}"
    f"{'SDNN':<12}"
    f"{'RMSSD':<12}"
    f"{'Invalid RR %':<12}"
)

print("-" * 100)

for r in results:

    print(
        f"{r['subject']:<10}"
        f"{r['window_id']:<10}"
        f"{r['num_peaks']:<10}"
        f"{r['mean_hr']:<12.2f}"
        f"{r['min_hr']:<12.2f}"
        f"{r['max_hr']:<12.2f}"
        f"{r['sdnn']:<12.4f}"
        f"{r['rmssd']:<12.4f}"
        f"{r['invalid_rr_percentage']:<12.2f}"
    )


# ============================================================
# SUMMARY
# ============================================================

if results:

    mean_hr_all = np.mean([r["mean_hr"] for r in results])
    mean_sdnn_all = np.mean([r["sdnn"] for r in results])
    mean_rmssd_all = np.mean([r["rmssd"] for r in results])
    mean_invalid_all = np.mean(
        [r["invalid_rr_percentage"] for r in results]
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Subjects validated       : {len(results)}")
    print(f"Average mean HR          : {mean_hr_all:.2f} BPM")
    print(f"Average SDNN             : {mean_sdnn_all:.4f} s")
    print(f"Average RMSSD            : {mean_rmssd_all:.4f} s")
    print(f"Average invalid RR       : {mean_invalid_all:.2f}%")

else:

    print("\nNo valid results obtained.")