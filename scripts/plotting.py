import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_ecg_segment(signal, annotations, fs, start_sec = 0, duration_sec = 10):
    start_idx = int(start_sec * fs)
    end_idx = int((start_sec + duration_sec) * fs)

    times = np.arange(start_idx, end_idx) / fs
    segment = signal[start_idx:end_idx, 0] # lead 0

    fig, ax = plt.subplots(figsize = (15,4))
    ax.plot(times, segment, label = "ECG Signal")

    # overlay annotations within this window
    for sample, symbol in zip(annotations["samples"], annotations["symbols"]):
        if start_idx <= sample < end_idx:
            time = sample / fs
            ax.axvline(x = time, color = 'r', linestyle = '--', alpha = 0.6)
            ax.text(time, max(segment), symbol, color = 'r', fontsize = 10, rotation = 90, verticalalignment = 'bottom')

    ax.set_title(f"ECG Signal with Annotations ({duration_sec} sec from {start_sec} sec)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_frequency_spectrum(sig, fs, max_freq = 1000, channel = 0, title = "Frequency Spectrum"):
    """
    Plot the frequency spectrum of on ECG channel to inspect powerline noise artifacts
    """
    
    n = len(sig)
    yf = fft(sig[:, channel])
    xf = fftfreq(n, 1 / fs)

    mask = (xf >= 0) & (xf <= max_freq)

    plt.figure(figsize = (12, 4))
    plt.plot(xf[mask], np.abs(yf[mask]))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_label_distribution(label_counts, title = "Beat Annotation Frequencies"):
    """
    Plot the distribution of beat labels using a dictionary of label_counts
    """

    # Dictionary for symbol explanations
    ANNOTATION_LEGEND = {
    "N": "Normal beat",
    "L": "Left bundle branch block beat",
    "R": "Right bundle branch block beat",
    "A": "Atrial premature beat",
    "V": "Premature ventricular contraction",
    "F": "Fusion of ventricular and normal beat",
    "e": "Atrial escape beat",
    "E": "Ventricular escape beat",
    "j": "Nodal (junctional) escape beat",
    "/": "Paced beat",
    "f": "Fusion of paced and normal beat",
    "a": "Aberrated atrial premature beat",
    "J": "Nodal (junctional) premature beat",
    "S": "Supraventricular premature beat"
}
    symbols = list(label_counts.keys())
    counts = list(label_counts.values())
    descriptions = [ANNOTATION_LEGEND.get(sym, "Unknown") for sym in symbols]

    plt.figure(figsize = (10, 4))
    bars = plt.bar(symbols, counts, color = "mediumseagreen")

    # Add label counts on top of bars
    for bar, desc, count in zip(bars, descriptions, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{count}", ha = 'center', va = 'bottom', fontsize = 9)

    # build legend
    legend_labels = [f"{sym}: {desc}" for sym, desc in zip(symbols, descriptions)]
    plt.legend(bars, legend_labels, title = "Beat Types", bbox_to_anchor = (1.05, 1), loc = "upper left", fontsize = 9)

    plt.title(title)
    plt.xlabel("Annotation Symbol")
    plt.ylabel("Count")
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.6)
    plt.tight_layout()
    plt.show()


def plot_pca_embedding(segments, labels, n_components = 2, max_points = 2000):
    """
    Flatten the beats and visualize PCA projection with labels
    """

    # flatten (n_beats, window_samples, n_channels) -> (n_beats, features)
    n_beats, window_size, n_channels = segments.shape
    flat_segments = segments.reshape(n_beats, -1)

    # subsample if too large for visualization
    if n_beats > max_points:
        np.random.seed(42)
        idx = np.random.choice(n_beats, max_points, replace = False)
        flat_segments = flat_segments[idx]
        labels = [labels[i] for i in idx]

    # standardize
    flat_segments = StandardScaler().fit_transform(flat_segments)

    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flat_segments)

    # plot
    plt.figure(figsize = (10, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                          c = [hash(1) % 20 for l in labels],
                          cmap = 'tab20',
                          alpha = 0.6,
                          s = 10
    )
    plt.title("PCA of ECG Beat Segments")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.tight_layout()

    # custom legend
    unique_labels = sorted(set(labels))
    handles = [plt.Line2D([0], [0], marker = 'o', color = 'w',
                          markerfacecolor = scatter.cmap(hash(l) % 20),
                          markersize = 6, label = l) for l in unique_labels]
    plt.legend(handles = handles, title = "Beat Types", bbox_to_anchor = (1.05, 1), loc = "upper left")
    plt.show()