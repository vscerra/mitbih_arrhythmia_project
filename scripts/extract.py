import numpy as np
from scripts.utils import get_project_path
from scripts.load_data import load_record
from scripts.preprocess import bandpass_filter, notch_filter, normalize, segment_beats, build_rr_lookup

def extract_beat_level_info(record_id, data_path = "data/raw", fs_override = None):
    """
    Process a full record and returns aligned with RR intervals, labels, and sample indices
    """
    record_data = load_record(record_id, data_path = data_path)
    signal = record_data["signal"]
    annotations = record_data["annotations"]
    fs = record_data["fs"] if fs_override is None else fs_override

    # preprocess signal
    filtered = bandpass_filter(signal, fs)
    filtered = notch_filter(filtered, fs)
    normalized = normalize(filtered)

    # segment beats and get annotations
    segments, labels, indices = segment_beats(normalized, annotations, fs)

    # build rr interval mapping
    rr_lookup = build_rr_lookup(annotations, fs)
    rr_intervals = [rr_lookup.get(idx, np.nan) for idx in indices]

    return{
        "record_id": record_id,
        "segments": segments,
        "labels": labels,
        "rr_intervals": rr_intervals,
        "sample_indices": indices
    }