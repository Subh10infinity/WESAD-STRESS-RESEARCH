import pickle
from pathlib import Path
import numpy as np


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

print("Loading WESAD S2...")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")

signals = data["signal"]["chest"]
labels = data["label"]

print("Data loaded.")
print("Subject:", data["subject"])


# ============================================================
# SELECT SIGNALS
# ============================================================

ecg = signals["ECG"].flatten()
eda = signals["EDA"].flatten()
temperature = signals["Temp"].flatten()
acc = signals["ACC"]


# ============================================================
# CREATE WINDOWS
# ============================================================

num_windows = len(labels) // WINDOW_SIZE

print("\n========== WINDOWING ==========")
print("Sampling rate:", SAMPLE_RATE, "Hz")
print("Window duration:", WINDOW_SECONDS, "seconds")
print("Samples per window:", WINDOW_SIZE)
print("Possible complete windows:", num_windows)


# ============================================================
# CREATE DATASET
# ============================================================

X = []
y = []

stress_windows = 0
nonstress_windows = 0
excluded_windows = 0

for i in range(num_windows):

    start = i * WINDOW_SIZE
    end = start + WINDOW_SIZE

    window_labels = labels[start:end]

    # --------------------------------------------------------
    # Stress = label 2
    # Non-stress = labels 1, 3, 4
    # Everything else = excluded
    # --------------------------------------------------------

    unique_labels = np.unique(window_labels)

    if np.all(window_labels == 2):

        target = 1
        stress_windows += 1

    elif np.all(np.isin(window_labels, [1, 3, 4])):

        target = 0
        nonstress_windows += 1

    else:

        excluded_windows += 1
        continue

    # --------------------------------------------------------
    # Store raw window
    # --------------------------------------------------------

    feature_window = {
        "ECG": ecg[start:end],
        "EDA": eda[start:end],
        "Temp": temperature[start:end],
        "ACC": acc[start:end]
    }

    X.append(feature_window)
    y.append(target)


# ============================================================
# SUMMARY
# ============================================================

print("\n========== WINDOWING RESULTS ==========")

print("Total complete windows:", num_windows)
print("Stress windows:", stress_windows)
print("Non-stress windows:", nonstress_windows)
print("Excluded windows:", excluded_windows)

print("\nTotal usable windows:", len(X))

if len(y) > 0:

    unique, counts = np.unique(y, return_counts=True)

    print("\nFinal class distribution:")

    for label, count in zip(unique, counts):

        class_name = "Stress" if label == 1 else "Non-stress"

        print(
            f"{class_name}: "
            f"{count} windows "
            f"({count / len(y) * 100:.2f}%)"
        )   