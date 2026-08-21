import mne


def filter_eeg(raw, l_freq=8.0, h_freq=30.0):
    """Band-pass filter EEG data for motor imagery."""
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    return raw
