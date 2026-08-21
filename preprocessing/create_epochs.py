from pathlib import Path
import mne

# Project root
project_root = Path(__file__).resolve().parent.parent

# EEG file
file_path = project_root / "data" / "eegmmidb" / "S001" / "S001R04.edf"

# Load EEG
raw = mne.io.read_raw_edf(file_path, preload=True)

# Read events
events, event_id = mne.events_from_annotations(raw)

print("Event IDs:", event_id)

# Keep only motor imagery events
selected_events = {
    "Left": event_id["T1"],
    "Right": event_id["T2"]
}

# Create epochs (0 to 4 seconds after cue)
epochs = mne.Epochs(
    raw,
    events,
    event_id=selected_events,
    tmin=0,
    tmax=4,
    baseline=None,
    preload=True
)

print(epochs)
print("Number of epochs:", len(epochs))