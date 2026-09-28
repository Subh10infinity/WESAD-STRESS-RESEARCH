import pickle
from pathlib import Path
import numpy as np

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# WESAD subject file
file_path = BASE_DIR / "data" / "WESAD" / "WESAD" / "S2" / "S2.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("========== WESAD DATASET INSPECTION ==========\n")

# ------------------------------------------------
# 1. Top-level information
# ------------------------------------------------

print("Top-level keys:")
print(data.keys())

print("\nSubject:")
print(data["subject"])

# ------------------------------------------------
# 2. Label information
# ------------------------------------------------

labels = data["label"]

print("\n========== LABEL INFORMATION ==========")
print("Label type:", type(labels))
print("Number of labels:", len(labels))

unique_labels, counts = np.unique(labels, return_counts=True)

print("\nUnique labels:")
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")

# ------------------------------------------------
# 3. Signal information
# ------------------------------------------------

signal = data["signal"]

print("\n========== SIGNAL INFORMATION ==========")
print("Signal type:", type(signal))

print("\nSignal keys:")
print(signal.keys())

# ------------------------------------------------
# 4. Signal shapes
# ------------------------------------------------

print("\n========== SIGNAL SHAPES ==========")

for location, sensors in signal.items():

    print(f"\n{location}:")

    for sensor, values in sensors.items():

        print(
            f"  {sensor}: "
            f"shape={values.shape}, "
            f"dtype={values.dtype}"
        )
print("\n========== LABEL MEANINGS ==========")

label_meanings = {
    0: "Not defined / transient",
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation",
    5: "Ignore",
    6: "???",
    7: "???"
}

for label in unique_labels:
    print(f"Label {label}: {label_meanings.get(label, 'Unknown')}")