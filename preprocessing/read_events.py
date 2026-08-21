from pathlib import Path
import mne

project_root = Path(__file__).resolve().parent.parent

file_path = project_root / "data" / "eegmmidb" / "S001" / "S001R04.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

events, event_id = mne.events_from_annotations(raw)

print("Events:")
print(events)

print("\nEvent IDs:")
print(event_id)