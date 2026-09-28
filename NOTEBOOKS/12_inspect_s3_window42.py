import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

FS = 700
WINDOW_SECONDS = 60
WINDOW_SIZE = FS * WINDOW_SECONDS

SUBJECT = "S3"
WINDOW_ID = 42


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "WESAD"
    / "WESAD"
    / SUBJECT
    / f"{SUBJECT}.pkl"
)


# ============================================================
# LOAD
# ============================================================

print(f"Loading {SUBJECT}...")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")


chest = data["signal"]["chest"]

ecg = chest["ECG"].flatten()
eda = chest["EDA"].flatten()
temp = chest["Temp"].flatten()
acc = chest["ACC"]

labels = np.asarray(data["label"])


# ============================================================
# WINDOW
# ============================================================

start = WINDOW_ID * WINDOW_SIZE
end = start + WINDOW_SIZE

ecg_window = ecg[start:end]
eda_window = eda[start:end]
temp_window = temp[start:end]
acc_window = acc[start:end]

label_window = labels[start:end]


print("\nWindow information:")
print("Subject:", SUBJECT)
print("Window:", WINDOW_ID)
print("Start sample:", start)
print("End sample:", end)

print("\nLabels:")
print(
    np.unique(
        label_window,
        return_counts=True
    )
)


# ============================================================
# ACCELEROMETER MAGNITUDE
# ============================================================

acc_magnitude = np.sqrt(
    np.sum(
        acc_window ** 2,
        axis=1
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\nSignal statistics:")

print(
    f"Temperature mean : {np.mean(temp_window):.4f}"
)

print(
    f"Temperature std  : {np.std(temp_window):.4f}"
)

print(
    f"EDA mean         : {np.mean(eda_window):.4f}"
)

print(
    f"EDA std          : {np.std(eda_window):.4f}"
)

print(
    f"ACC mean         : {np.mean(acc_magnitude):.4f}"
)

print(
    f"ACC std          : {np.std(acc_magnitude):.4f}"
)


# ============================================================
# PLOT TEMPERATURE
# ============================================================

time_temp = np.arange(
    len(temp_window)
) / FS

plt.figure(figsize=(14, 5))

plt.plot(
    time_temp,
    temp_window
)

plt.xlabel("Time (seconds)")
plt.ylabel("Temperature")

plt.title(
    "S3 Window 42 - Temperature"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT ACCELEROMETER
# ============================================================

# ACC has the same 700 Hz sampling rate
time_acc = np.arange(
    len(acc_magnitude)
) / FS

plt.figure(figsize=(14, 5))

plt.plot(
    time_acc,
    acc_magnitude
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration Magnitude")

plt.title(
    "S3 Window 42 - Accelerometer Magnitude"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT EDA
# ============================================================

time_eda = np.arange(
    len(eda_window)
) / FS

plt.figure(figsize=(14, 5))

plt.plot(
    time_eda,
    eda_window
)

plt.xlabel("Time (seconds)")
plt.ylabel("EDA")

plt.title(
    "S3 Window 42 - EDA"
)

plt.grid(True)

plt.tight_layout()

plt.show()


print("\nInspection complete.")