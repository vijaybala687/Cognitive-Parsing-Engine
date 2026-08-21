from pathlib import Path
import numpy as np
import mne

# -----------------------------
# Project paths
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "eegmmidb"
processed_path = project_root / "processed"
processed_path.mkdir(exist_ok=True)

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
        raw = mne.io.read_raw_edf(
            edf_file,
            preload=True,
            verbose=False
        )

        # -----------------------------
        # Band-pass filter
        # -----------------------------
        raw.filter(
            8.,
            30.,
            fir_design="firwin",
            verbose=False
        )

        # -----------------------------
        # Read events
        # -----------------------------
        events, event_id = mne.events_from_annotations(raw)

        if "T1" not in event_id or "T2" not in event_id:
            print("Skipping - T1/T2 not found")
            continue

        event_dict = {
            "Left": event_id["T1"],
            "Right": event_id["T2"]
        }

        # -----------------------------
        # Create epochs
        # -----------------------------
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=0,
            tmax=4,
            baseline=None,
            preload=True,
            verbose=False
        )

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

# -----------------------------
# Combine all subjects
# -----------------------------
print("\nCombining all subjects...")

X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)

print("\nFinal Dataset")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# Save dataset
# -----------------------------
np.save(processed_path / "X.npy", X)
np.save(processed_path / "y.npy", y)

print("\nDataset saved successfully!")
print("Saved X to:", processed_path / "X.npy")
print("Saved y to:", processed_path / "y.npy")