import matplotlib.pyplot as plt
import numpy as np

def plot_wavelet_heatmap(W, title='Wavelet Coefficient Heatmap', save_path=None):
    """
    Plot a wavelet coefficient matrix as a heatmap.
    
    Args:
        W (np.ndarray): 2D array of wavelet coefficients.
        title (str): Title for the plot.
        save_path (str): Path to save the figure. If None, display the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(W), aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(label='|Coefficient|')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()