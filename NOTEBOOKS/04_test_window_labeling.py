import pickle
from pathlib import Path
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 700
WINDOW_SECONDS = 60
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

MAJORITY_THRESHOLD = 0.80

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

labels = data["label"]

print("Subject:", data["subject"])


# ============================================================
# LABEL DEFINITIONS
# ============================================================

STRESS_LABEL = 2

NON_STRESS_LABELS = [1, 3, 4]

VALID_LABELS = [1, 2, 3, 4]


# ============================================================
# WINDOWING
# ============================================================

num_windows = len(labels) // WINDOW_SIZE

stress_windows = 0
nonstress_windows = 0
excluded_windows = 0

print("\n========== TESTING 80% MAJORITY RULE ==========")

for i in range(num_windows):

    start = i * WINDOW_SIZE
    end = start + WINDOW_SIZE

    window_labels = labels[start:end]

    unique, counts = np.unique(
        window_labels,
        return_counts=True
    )

    distribution = dict(zip(unique, counts))

    # Remove labels that we don't want to classify
    valid_counts = {
        label: distribution.get(label, 0)
        for label in VALID_LABELS
    }

    # Find dominant valid label
    dominant_label = max(
        valid_counts,
        key=valid_counts.get
    )

    dominant_count = valid_counts[dominant_label]

    dominance_ratio = dominant_count / WINDOW_SIZE

    # --------------------------------------------------------
    # Apply majority rule
    # --------------------------------------------------------

    if dominance_ratio >= MAJORITY_THRESHOLD:

        if dominant_label == STRESS_LABEL:

            stress_windows += 1

        elif dominant_label in NON_STRESS_LABELS:

            nonstress_windows += 1

    else:

        excluded_windows += 1


# ============================================================
# RESULTS
# ============================================================

usable = stress_windows + nonstress_windows

print("\n========== RESULTS ==========")

print("Total windows:", num_windows)

print("Stress windows:", stress_windows)

print("Non-stress windows:", nonstress_windows)

print("Excluded windows:", excluded_windows)

print("Usable windows:", usable)

if usable > 0:

    print("\nClass distribution:")

    print(
        f"Stress: "
        f"{stress_windows} "
        f"({stress_windows / usable * 100:.2f}%)"
    )

    print(
        f"Non-stress: "
        f"{nonstress_windows} "
        f"({nonstress_windows / usable * 100:.2f}%)"
    )