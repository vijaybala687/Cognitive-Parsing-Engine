import mne


def create_epochs(raw, events, event_id, tmin=0, tmax=4, resample_rate=160):
    """Create epochs from event annotations and optionally resample them."""
    selected_events = {
        "Left": event_id["T1"],
        "Right": event_id["T2"],
    }

    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_events,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    if resample_rate is not None:
        epochs.resample(resample_rate)

    return epochs
