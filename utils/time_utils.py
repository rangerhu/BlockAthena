import numpy as np

def bin_timestamps(timestamps, bins=100):
    """Bin timestamps into frequency counts across a fixed number of intervals."""
    timestamps = np.array(timestamps)
    min_ts, max_ts = np.min(timestamps), np.max(timestamps)
    bin_edges = np.linspace(min_ts, max_ts, bins + 1)
    binned_counts, _ = np.histogram(timestamps, bins=bin_edges)
    return binned_counts, bin_edges