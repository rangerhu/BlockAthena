import os
import numpy as np
import networkx as nx
import argparse
import logging

from mpm.motif_modeler import run_motif_extraction
from mpm.wavelet_analyzer import run_wavelet_analysis
from mpm.subgraph_segmenter import build_transaction_graph, segment_by_periods, save_subgraphs
from behavior.mixed_order_analyzer import analyze_mixed_order_features
from era.era_aggregator import gated_fusion, weighted_aggregation, detect_eth_crime
from utils.time_utils import bin_timestamps
from utils.io_utils import ensure_dir, save_pickle

# Setup logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def compute_wavelet_energy(coeffs, peaks):
    """Compute spectral energy of each evolution period from wavelet coefficients."""
    energy_list = []
    for scale_idx in peaks:
        energy = np.sum(np.abs(coeffs[scale_idx])**2)
        energy_list.append(energy)
    return np.array(energy_list)

def main(tx_csv, target, output_dir):
    ensure_dir(output_dir)
    output_target_dir = os.path.join(output_dir, target)
    ensure_dir(output_target_dir)

    logging.info("Step 1: Building transaction graph")
    tx_graph = build_transaction_graph(tx_csv)
    timestamps = [d['timestamp'] for _, _, d in tx_graph.edges(data=True)]

    logging.info("Step 2: Running motif extraction and wavelet analysis")
    run_motif_extraction(tx_csv, os.path.join(output_target_dir, 'motifs.txt'))
    series, _ = bin_timestamps(timestamps, bins=100)
    coeffs, peaks, _ = run_wavelet_analysis(series, os.path.join(output_target_dir, 'wavelet_heatmap.png'))

    logging.info("Step 3: Segmenting by evolution periods")
    subgraphs = segment_by_periods(tx_graph, timestamps, peaks.tolist())
    save_subgraphs(subgraphs, output_target_dir)

    logging.info("Step 4: Analyzing mixed-order features (DDGM + CHGM)")
    analyze_mixed_order_features(target, subgraphs)

    logging.info("Step 5: Fusing features and aggregating long-term behavior")
    ddgm = np.load(f'data/processed/{target}/ddgm.npy')
    chgm = np.load(f'data/processed/{target}/chgm.npy')
    wavelet_energy = compute_wavelet_energy(coeffs, peaks)
    fused = gated_fusion(ddgm, chgm)
    Z_final = weighted_aggregation(fused, wavelet_energy)

    logging.info("Step 6: Detecting Ethereum crime")
    y_binary = 1  # placeholder
    y_multi = 2
    result, category = detect_eth_crime(Z_final, y_binary, y_multi)
    logging.info(f"[RESULT] Criminal activity: {result} | Category: {category}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BlockAthena pipeline')
    parser.add_argument('--input', type=str, default='data/raw/transactions.csv', help='Path to transaction CSV')
    parser.add_argument('--target', type=str, default='example', help='Target label (e.g., case ID)')
    parser.add_argument('--outdir', type=str, default='data/processed', help='Output directory')
    args = parser.parse_args()

    main(args.input, args.target, args.outdir)