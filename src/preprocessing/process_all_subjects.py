from pathlib import Path

import numpy as np

from src.features.extractors import extract_csp_features, save_csp_features
from src.preprocessing.epoching import create_epochs
from src.preprocessing.filters import filter_eeg
from src.preprocessing.loader import load_eeg_data
from src.preprocessing.read_events import read_event_annotations


# -----------------------------
# Project paths
# -----------------------------
project_root = Path(__file__).resolve().parents[2]
dataset_path = project_root / "data" / "eegmmidb"
processed_path = project_root / "data" / "processed"
processed_path.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Subject folders
# -----------------------------
subjects = sorted(
    [folder for folder in dataset_path.iterdir() if folder.is_dir()]
)

print(f"Found {len(subjects)} subjects")

# Motor imagery runs
runs = ["R04", "R08", "R12"]

all_data = []
all_labels = []
all_csp_features = []
all_csp_labels = []

# -----------------------------
# Process every subject
# -----------------------------
for subject in subjects:

    print(f"\nProcessing {subject.name}")

    for run in runs:

        edf_file = subject / f"{subject.name}{run}.edf"

        if not edf_file.exists():
            print(f"Missing {edf_file.name}")
            continue

        print(f"Loading {edf_file.name}")

        # -----------------------------
        # Load EEG
        # -----------------------------
        raw = load_eeg_data(edf_file)

        # -----------------------------
        # Band-pass filter
        # -----------------------------
        raw = filter_eeg(raw)

        # -----------------------------
        # Read events
        # -----------------------------
        events, event_id = read_event_annotations(raw)

        if "T1" not in event_id or "T2" not in event_id:
            print("Skipping - T1/T2 not found")
            continue

        # -----------------------------
        # Create epochs
        # -----------------------------
        epochs = create_epochs(raw, events, event_id)

        # -----------------------------
        # Make all recordings same sampling rate
        # -----------------------------
        epochs.resample(160)

        # -----------------------------
        # Extract data
        # -----------------------------
        X = epochs.get_data()
        y = epochs.events[:, -1]

        print(f"Epoch shape: {X.shape}")

        all_data.append(X)
        all_labels.append(y)

        X_csp, y_csp = extract_csp_features(raw, events, event_id)
        all_csp_features.append(X_csp)
        all_csp_labels.append(y_csp)

# -----------------------------
# Combine all subjects
# -----------------------------
print("\nCombining all subjects...")

X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)

X_features = np.concatenate(all_csp_features, axis=0) if all_csp_features else X
y_labels = np.concatenate(all_csp_labels, axis=0) if all_csp_labels else y

print("\nFinal Dataset")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_features shape:", X_features.shape)
print("y_labels shape:", y_labels.shape)

# -----------------------------
# Save dataset
# -----------------------------
np.save(processed_path / "X.npy", X)
np.save(processed_path / "y.npy", y)
save_csp_features(X_features, y_labels, processed_path)

print("\nDataset saved successfully!")
print("Saved X to:", processed_path / "X.npy")
print("Saved y to:", processed_path / "y.npy")
print("Saved X_features to:", processed_path / "X_features.npy")
print("Saved y_labels to:", processed_path / "y_labels.npy")
