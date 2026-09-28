import pickle
from pathlib import Path
import numpy as np

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset path
file_path = (
    BASE_DIR
    / "data"
    / "WESAD"
    / "WESAD"
    / "S2"
    / "S2.pkl"
)

# Load data
with open(file_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

labels = data["label"]

print("========== WESAD LABEL INSPECTION ==========\n")

print("Subject:", data["subject"])
print("Total samples:", len(labels))

print("\nUnique labels:")

unique_labels, counts = np.unique(labels, return_counts=True)

for label, count in zip(unique_labels, counts):
    percentage = (count / len(labels)) * 100

    print(
        f"Label {label}: "
        f"{count:,} samples "
        f"({percentage:.2f}%)"
    )

print("\n========== LABEL TRANSITIONS ==========\n")

# Find points where label changes
changes = np.where(labels[1:] != labels[:-1])[0] + 1

print("Number of label transitions:", len(changes))

print("\nFirst 50 transitions:")

for i, index in enumerate(changes[:50]):

    previous_label = labels[index - 1]
    new_label = labels[index]

    time_seconds = index / 700

    print(
        f"{i + 1:02d}. "
        f"{previous_label} → {new_label} "
        f"at sample {index:,} "
        f"({time_seconds:.2f} seconds)"
    )