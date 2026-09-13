import mne
import numpy as np
from pathlib import Path
import warnings

# Suppress the harmless MNE annotation truncation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths to your local raw dataset
data_dir = Path("data/eegmmidb")
out_dir = Path("data/processed")
X_list, y_list = [], []

print("Extracting 4-Class Motor Imagery Data (Left, Right, Up, Down)...")
print("Filtering out EOF truncated artifacts...")

for subj_dir in sorted(data_dir.glob("S*")):
    # PhysioNet Runs: 
    # 4, 8, 12 = Left/Right  |  6, 10, 14 = Both Fists(Up)/Both Feet(Down)
    for run in [4, 8, 12, 6, 10, 14]:
        file_path = subj_dir / f"{subj_dir.name}R{run:02d}.edf"
        if not file_path.exists(): continue
        
        try:
            # Load raw EEG and isolate annotations
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            events, _ = mne.events_from_annotations(raw, event_id={'T1': 2, 'T2': 3}, verbose=False)
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            
            # Slice into exactly 4.0 seconds
            epochs = mne.Epochs(raw, events, tmin=0, tmax=4.0, picks=picks, baseline=None, preload=True, verbose=False)
            
            for i, event in enumerate(epochs.events):
                code = event[2] 
                label = -1
                
                if run in [4, 8, 12]:
                    if code == 2: label = 0   # Class 0: Left Hand
                    elif code == 3: label = 1 # Class 1: Right Hand
                elif run in [6, 10, 14]:
                    if code == 2: label = 2   # Class 2: Both Fists (Up)
                    elif code == 3: label = 3 # Class 3: Both Feet (Down)
                
                if label != -1:
                    epoch_data = epochs.get_data(copy=False)[i]
                    
                    # STRICT ENFORCEMENT: Only append if the shape is perfectly (64, 641)
                    if epoch_data.shape == (64, 641):
                        X_list.append(epoch_data)
                        y_list.append(label)
        except Exception:
            pass

X = np.array(X_list).astype(np.float32)
y = np.array(y_list).astype(np.int64)

print(f"\nExtraction Complete! Successfully extracted {len(X)} perfect epochs.")
print(f"X shape: {X.shape} | y shape: {y.shape}")

# Save the new massive 4-class arrays
np.save(out_dir / "X.npy", X)
np.save(out_dir / "y.npy", y)