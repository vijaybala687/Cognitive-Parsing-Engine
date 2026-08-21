from pathlib import Path
import mne
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent

file_path = project_root / "data" / "eegmmidb" / "S001" / "S001R04.edf"

raw = mne.io.read_raw_edf(file_path, preload=True)

print(raw)

fig = raw.plot(duration=10, n_channels=20, scalings="auto", block=False)

plt.show()