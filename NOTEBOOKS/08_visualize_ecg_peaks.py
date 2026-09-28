import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
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
labels = np.asarray(data["label"])

print("ECG samples:", len(ecg))


# ============================================================
# FIND FIRST PURE STRESS WINDOW
# ============================================================

stress_window = None

for window_id in range(len(ecg) // WINDOW_SIZE):

    start = window_id * WINDOW_SIZE
    end = start + WINDOW_SIZE

    window_labels = labels[start:end]

    if np.all(window_labels == 2):

        stress_window = window_id

        break


if stress_window is None:
    raise RuntimeError("No pure stress window found.")


start = stress_window * WINDOW_SIZE
end = start + WINDOW_SIZE

ecg_window = ecg[start:end]

print("Stress window:", stress_window)
print("Start sample:", start)
print("End sample:", end)


# ============================================================
# BANDPASS FILTER
# ============================================================

nyquist = 0.5 * FS

low = LOWCUT / nyquist
high = HIGHCUT / nyquist

b, a = butter(
    FILTER_ORDER,
    [low, high],
    btype="band"
)

filtered_ecg = filtfilt(
    b,
    a,
    ecg_window
)


# ============================================================
# R-PEAK DETECTION
# ============================================================

threshold = (
    np.mean(filtered_ecg)
    + 0.5 * np.std(filtered_ecg)
)

peaks, properties = find_peaks(
    filtered_ecg,
    height=threshold,
    distance=int(0.30 * FS),
    prominence=0.5 * np.std(filtered_ecg)
)

print("Detected peaks:", len(peaks))


# ============================================================
# PLOT FIRST 10 SECONDS
# ============================================================

plot_seconds = 10

plot_samples = plot_seconds * FS

time = np.arange(plot_samples) / FS

plt.figure(figsize=(14, 6))

plt.plot(
    time,
    filtered_ecg[:plot_samples],
    label="Filtered ECG"
)

visible_peaks = peaks[peaks < plot_samples]

plt.scatter(
    visible_peaks / FS,
    filtered_ecg[visible_peaks],
    marker="o",
    label="Detected R-peaks"
)

plt.xlabel("Time (seconds)")
plt.ylabel("ECG amplitude")

plt.title(
    f"S2 Stress Window {stress_window} - ECG R-Peak Detection"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()