from pathlib import Path

import mne
import numpy as np
from mne.decoding import CSP

from src.preprocessing.epoching import create_epochs
from src.preprocessing.filters import filter_eeg
from src.preprocessing.loader import load_eeg_data
from src.preprocessing.read_events import read_event_annotations, select_motor_imagery_events


def get_csp_features(file_path=None):
    """Load EEG data, preprocess it, and return CSP-transformed features and labels."""
    if file_path is None:
        project_root = Path(__file__).resolve().parents[2]
        file_path = project_root / "data" / "eegmmidb" / "S001" / "S001R04.edf"

    raw = load_eeg_data(file_path)
    raw = filter_eeg(raw)
    events, event_id = read_event_annotations(raw)

    if "T1" not in event_id or "T2" not in event_id:
        raise KeyError("T1/T2 event annotations are required for left/right motor imagery.")

    epochs = create_epochs(raw, events, event_id)
    X = epochs.get_data()
    y = epochs.events[:, -1]

    csp = CSP(n_components=4, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)

    return X_csp, y


def extract_csp_features(raw, events, event_id):
    """Extract CSP features from preprocessed raw data and events."""
    selected_events = select_motor_imagery_events(event_id)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_events,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    csp = CSP(n_components=4, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)

    return X_csp, y


def save_csp_features(X_features, y_labels, processed_dir="data/processed"):
    """Save extracted CSP features and labels to the processed dataset folder."""
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    np.save(processed_path / "X_features.npy", X_features)
    np.save(processed_path / "y_labels.npy", y_labels)
    return processed_path / "X_features.npy", processed_path / "y_labels.npy"


if __name__ == "__main__":
    X_csp, y = get_csp_features()
    save_csp_features(X_csp, y)

    print("CSP feature shape:", X_csp.shape)
    print("Labels:", y)
    print("CSP Features:")
    print(X_csp)
