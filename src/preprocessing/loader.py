from pathlib import Path

import mne


def load_eeg_data(file_path):
    """Load an EDF file as an MNE Raw object."""
    return mne.io.read_raw_edf(file_path, preload=True, verbose=False)


def load_subject_file(project_root, subject_name, run_name):
    """Return the EDF path for a subject/run pair."""
    return Path(project_root) / "data" / "eegmmidb" / subject_name / f"{subject_name}{run_name}.edf"
