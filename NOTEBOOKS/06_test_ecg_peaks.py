import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 700
WINDOW_SECONDS = 60
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "WESAD"
    / "WESAD"
    / "S2"
    / "S2.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading S2...")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")

ecg = data["signal"]["chest"]["ECG"].flatten()
labels = data["label"]

print("Data loaded.")
print("ECG samples:", len(ecg))


# ============================================================
# FIND FIRST PURE STRESS WINDOW
# ============================================================

print("\nSearching for a stress window...")

stress_window = None

num_windows = len(labels) // WINDOW_SIZE

for window_id in range(num_windows):

    start = window_id * WINDOW_SIZE
    end = start + WINDOW_SIZE

    window_labels = labels[start:end]

    # Use the same strict stress definition
    if np.all(window_labels == 2):

        stress_window = ecg[start:end]

        print("Found stress window:", window_id)
        print("Start sample:", start)
        print("End sample:", end)

        break


if stress_window is None:

    raise ValueError("No pure stress window found.")


# ============================================================
# BANDPASS FILTER
# ============================================================

print("\nApplying ECG bandpass filter...")

LOWCUT = 5.0
HIGHCUT = 20.0

nyquist = SAMPLE_RATE / 2

low = LOWCUT / nyquist
high = HIGHCUT / nyquist

b, a = butter(
    3,
    [low, high],
    btype="band"
)

filtered_ecg = filtfilt(
    b,
    a,
    stress_window
)


# ============================================================
# R-PEAK DETECTION
# ============================================================

print("Detecting R-peaks...")

# Estimate a robust threshold
threshold = (
    np.mean(filtered_ecg)
    + 0.5 * np.std(filtered_ecg)
)

# Minimum distance between heartbeats
# 0.30 seconds ≈ maximum 200 BPM

minimum_distance = int(
    0.30 * SAMPLE_RATE
)

peaks, properties = find_peaks(
    filtered_ecg,
    height=threshold,
    distance=minimum_distance,
    prominence=0.5 * np.std(filtered_ecg)
)


# ============================================================
# CALCULATE HEART RATE
# ============================================================

rr_intervals = np.diff(peaks) / SAMPLE_RATE

if len(rr_intervals) == 0:

    raise ValueError(
        "No valid RR intervals detected."
    )

heart_rate = 60 / rr_intervals


# ============================================================
# RESULTS
# ============================================================

print("\n========== ECG PEAK DETECTION RESULTS ==========")

print("Window duration:", WINDOW_SECONDS, "seconds")

print("Detected R-peaks:", len(peaks))

print(
    "Average RR interval:",
    round(np.mean(rr_intervals), 4),
    "seconds"
)

print(
    "Average heart rate:",
    round(np.mean(heart_rate), 2),
    "BPM"
)

print(
    "Minimum heart rate:",
    round(np.min(heart_rate), 2),
    "BPM"
)

print(
    "Maximum heart rate:",
    round(np.max(heart_rate), 2),
    "BPM"
)

print(
    "RR interval standard deviation:",
    round(np.std(rr_intervals), 4),
    "seconds"
)

print("\nFirst 20 R-peak positions:")

print(peaks[:20])

print("\nFirst 20 RR intervals:")

print(rr_intervals[:20])