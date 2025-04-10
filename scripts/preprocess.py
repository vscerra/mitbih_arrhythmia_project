import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def bandpass_filter(sig, fs, lowcut = 0.5, highcut = 40.0, order = 4):
    """
    Applying a bandpass filter to remove exceptionally high and low frequencies
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype = "band")
    filtered = signal.filtfilt(b, a, sig, axis = 0)
    return filtered

def notch_filter(sig, fs, freq = 60.0, quality = 30.0):
    """
    Apply a notch filter to remove powerline interference at freq (usually 60 Hz in the US)
    """
    nyq = 0.5 * fs
    norm_freq = freq / nyq
    b, a = signal.iirnotch(norm_freq, quality)
    filtered = signal.filtfilt(b, a, sig, axis = 0)
    return filtered

def normalize(sig, method = "zscore"):
    """
    Normalize the signal channel-wise using the specified method. - method = "zscore" or "minmax"
    """
    if method == "zscore":
        return (sig - np.mean(sig, axis = 0) / np.std(sig, axis = 0))
    elif method == "minmax":
        return (sig - np.min(sig, axis = 0) / (np.max(sig, axis = 0) - np.min(sig, axis = 0)))
    else: 
        raise ValueError(f"Unknown normalization method: {method}")
    

def segment_beats(signal, annotations, fs, window_size_sec = 0.6, pre_peak_ratio = 0.33):
    """
    Extract fixed-length windows around each annotated R-peak.
    
    Parameters:
    - signal: np.ndarray of shape (n_samples, n_channels)
    - annotations: dict with 'samples' and 'symbols'
    - fs: sampling frequency
    - window_size_sec: total window duration in seconds
    - pre_peak_ratio: proportion of the window before the R-peak (eg. 0.33 means 1/3 before, 2/3 after)
    
    Returns:
    - segments: np.ndarray of shape (n_beats, window_samples, n_channels)
    - labels: list of annotation symbols
    """

    beat_samples = int(fs * window_size_sec)
    pre_peak = int(beat_samples * pre_peak_ratio)
    post_peak = beat_samples - pre_peak
    segments = []
    labels = []
    indices = []

    for i, peak_idx in enumerate(annotations["samples"]):
        start = peak_idx - pre_peak
        end = peak_idx + post_peak

        # skip segments that would go out of bounds
        if start < 0 or end > signal.shape[0]:
            continue

        segment = signal[start:end]
        segments.append(segment)
        labels.append(annotations["symbols"][i])
        indices.append(peak_idx)

    segments = np.stack(segments, axis = 0)

    return segments, labels, indices


def build_rr_lookup(annotations, fs):
    """
    Returns a dict mapping beat sample index -> RR interval
    """
    rr_lookup = {}
    samples = annotations["samples"]
    for i in range(1, len(samples)):
        rr = (samples[i] - samples[i - 1]) / fs
        rr_lookup[samples[i]] = rr # assign RR to the current beat

    return rr_lookup


def compute_rr_intervals(annotations, fs):
    """
    Compute RR intevals (interbeat intervals, in seconds) from annotation sample indices
    """
    sample_indices = np.array(annotations["samples"])
    rr_intervals = np.diff(sample_indices) / fs # in s
    # Pad with 0 or NaN to match number of beats
    rr_intervals = np.insert(rr_intervals, 0, np.nan)

    return rr_intervals
