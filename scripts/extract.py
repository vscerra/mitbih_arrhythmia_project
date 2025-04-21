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


def build_beat_sequences(segments, labels_df, sequence_length=5):
    """
    Construct sequences of beats ordered by record and sample index.
    Returns: X (sequence windows), y (center-beat label)
    """
    sequences = []
    targets = []
    abnormal_labels = {"A", "V", "~", "|", "Q", "+", "f"}
    normal_labels = {"N", "/", "f"}
    labels_df['label'] = labels_df['label'].apply(lambda x: "A" if x in abnormal_labels else "N").tolist()
    labels_df = labels_df.copy()
    

    for rid in labels_df["record"].unique():
        df_rec = labels_df[labels_df["record"] == rid].sort_values("sample_index")
        indices = df_rec["sample_index"].values
        labels = df_rec["label"].values

        for i in range(len(indices) - sequence_length + 1):
            seq_idx = indices[i:i + sequence_length]
            if np.any(seq_idx >= len(segments)):
                continue
            seq_seg = segments[seq_idx]
            label_seq = labels[i + sequence_length // 2] # label center beat

            sequences.append(seq_seg)
            targets.append(label_seq)

    X_seq = np.array(sequences)
    y_seq = np.array(targets)

    # Flatten each beat: (sequence_len, 216, 2)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], -1))
    
    return X_seq, y_seq