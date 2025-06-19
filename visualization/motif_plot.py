import matplotlib.pyplot as plt

def plot_motif_frequency(motif_counts, save_path=None):
    """
    Plot the frequency distribution of motifs.
    
    Args:
        motif_counts (dict): Dictionary mapping motif labels to counts.
        save_path (str): Optional path to save the figure.
    """
    labels = list(motif_counts.keys())
    counts = list(motif_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.title('Motif Frequency Distribution')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()