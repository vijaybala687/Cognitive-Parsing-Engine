from pathlib import Path
import mne

project_root = Path(__file__).resolve().parent.parent

file_path = project_root / "data" / "eegmmidb" / "S001" / "S001R04.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

print("Before filtering")
print(raw)

# Band-pass filter for motor imagery
raw.filter(l_freq=8, h_freq=30)

print("\nAfter filtering")
print(raw)