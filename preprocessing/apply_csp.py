from pathlib import Path
import mne
from mne.decoding import CSP


def get_csp_features():
    # Project root
    project_root = Path(__file__).resolve().parent.parent

    # EEG file
    file_path = project_root / "data" / "eegmmidb" / "S001" / "S001R04.edf"

    # Load EEG
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Filter for motor imagery
    raw.filter(8., 30., fir_design="firwin")

    # Read events
    events, event_id = mne.events_from_annotations(raw)

    # Keep only left and right motor imagery
    event_dict = {
        "Left": event_id["T1"],
        "Right": event_id["T2"]
    }

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True
    )

    # Extract data and labels
    X = epochs.get_data()
    y = epochs.events[:, -1]

    # Apply CSP
    csp = CSP(n_components=4, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)

    return X_csp, y


if __name__ == "__main__":
    X_csp, y = get_csp_features()

    print("CSP feature shape:", X_csp.shape)
    print("Labels:", y)
    print("CSP Features:")
    print(X_csp)