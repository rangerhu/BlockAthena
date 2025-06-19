import os
import networkx as nx
from collections import defaultdict
from utils.io_utils import ensure_dir, save_pickle
from config import DATA_DIR

def build_transaction_graph(tx_csv):
    """Build a directed graph from transaction CSV."""
    import pandas as pd
    df = pd.read_csv(tx_csv)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['from'], row['to'], timestamp=row['timestamp'], value=row['value'])
    return G

def segment_by_periods(graph, timestamps, period_peaks):
    """Split graph into period-oriented subgraphs based on wavelet peaks."""
    timestamps = sorted(timestamps)
    time_bounds = []

    for i in range(len(period_peaks)):
        if i == 0:
            start = timestamps[0]
        else:
            start = (timestamps[period_peaks[i - 1]] + timestamps[period_peaks[i]]) // 2
        end = timestamps[period_peaks[i]]
        time_bounds.append((start, end))

    subgraphs = defaultdict(list)
    for idx, (start, end) in enumerate(time_bounds):
        G_sub = nx.DiGraph()
        for u, v, d in graph.edges(data=True):
            ts = d.get('timestamp')
            if start <= ts <= end:
                G_sub.add_edge(u, v, **d)
        if len(G_sub.edges) > 0:
            subgraphs[f'period_{idx}'].append(G_sub)
    return subgraphs

def save_subgraphs(subgraphs, output_dir):
    """Save subgraphs as .gpickle files under output directory."""
    out_path = os.path.join(output_dir, 'subgraphs')
    ensure_dir(out_path)
    for period, graphs in subgraphs.items():
        for idx, g in enumerate(graphs):
            path = os.path.join(out_path, f'{period}_{idx}.gpickle')
            nx.write_gpickle(g, path)