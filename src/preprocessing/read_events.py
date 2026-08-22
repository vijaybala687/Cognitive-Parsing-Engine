import mne


def read_event_annotations(raw):
    """Read EEG annotations and return event array plus event IDs."""
    events, event_id = mne.events_from_annotations(raw)
    return events, event_id


def select_motor_imagery_events(event_id):
    """Return the selected motor imagery events for left and right cues."""
    if "T1" not in event_id or "T2" not in event_id:
        raise KeyError("T1/T2 event annotations are required for left/right motor imagery.")

    return {
        "Left": event_id["T1"],
        "Right": event_id["T2"],
    }
