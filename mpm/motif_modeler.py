import pandas as pd
import networkx as nx
from collections import defaultdict
from utils.io_utils import ensure_dir

def build_temporal_graph(tx_csv):
    """Load transaction CSV and construct a temporal MultiDiGraph."""
    df = pd.read_csv(tx_csv)
    df = df.sort_values(by='timestamp')
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['from'], row['to'], timestamp=row['timestamp'], value=row['value'])
    return G

def detect_motifs(graph):
    """Detect simple fan motifs (A → B, A → C) in the graph."""
    motifs = defaultdict(int)
    for node in graph.nodes:
        neighbors = list(graph.successors(node))
        if len(neighbors) >= 2:
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    motifs[(node, neighbors[i], neighbors[j])] += 1
    return motifs

def run_motif_extraction(tx_csv, output_path):
    """Run motif detection and save results to file."""
    print("[INFO] Running motif extraction...")
    G = build_temporal_graph(tx_csv)
    motifs = detect_motifs(G)
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        for motif, count in motifs.items():
            f.write(f"{motif[0]} -> {motif[1]}, {motif[0]} -> {motif[2]} : {count}\n")
    print(f"[INFO] Motif results saved to {output_path}")