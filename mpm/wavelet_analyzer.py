import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks
from config import WAVELET_TYPE, MIN_SCALE, MAX_SCALE, NUM_SCALES

def compute_wavelet_transform(signal, wavelet=WAVELET_TYPE):
    """Compute Continuous Wavelet Transform (CWT) on the input signal."""
    scales = np.linspace(MIN_SCALE, MAX_SCALE, NUM_SCALES)
    coef, freqs = pywt.cwt(signal, scales, wavelet)
    power = np.abs(coef) ** 2
    return power, freqs, scales

def identify_dominant_periods(power):
    """Identify peaks in the average wavelet power across time."""
    avg_power = np.mean(power, axis=1)
    peaks, _ = find_peaks(avg_power)
    return peaks, avg_power

def plot_wavelet_heatmap(power, output_path):
    """Plot and save a heatmap of the wavelet power spectrum."""
    plt.figure(figsize=(10, 6))
    plt.imshow(power, aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Wavelet Power')
    plt.title('Wavelet Time-Frequency Power Spectrum')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_wavelet_analysis(series, output_path):
    """Run full wavelet analysis and return dominant periods."""
    print("[INFO] Running wavelet transform...")
    power, freqs, scales = compute_wavelet_transform(series)
    peaks, avg_power = identify_dominant_periods(power)
    plot_wavelet_heatmap(power, output_path)
    print(f"[INFO] Wavelet heatmap saved to {output_path}")
    return peaks, freqs