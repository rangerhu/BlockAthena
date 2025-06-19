import os
import numpy as np
import networkx as nx
from utils.io_utils import ensure_dir, save_pickle

def extract_ddgm(subgraphs, q=2):
    """Extract DDGM low-order anomaly features from period subgraphs."""
    ddgm_features = []
    for period in sorted(subgraphs.keys()):
        for g in subgraphs[period]:
            collusive_scores = {}
            for v in g.nodes():
                edges = g.in_edges(v, data=True)
                timestamps = [d['timestamp'] for _, _, d in edges]
                collusive_scores[v] = max(timestamps) if timestamps else 0

            weights = {v: np.exp(score) for v, score in collusive_scores.items()}
            total = sum(weights.values())
            norm_weights = {v: w / total for v, w in weights.items()}

            node_feats = {v: np.random.rand(16) for v in g.nodes()}  # placeholder
            for _ in range(q):
                new_feats = {}
                for v in g.nodes():
                    neigh = g.predecessors(v)
                    agg = sum(norm_weights.get(u, 0) * node_feats[u] for u in neigh)
                    new_feats[v] = node_feats[v] + agg
                node_feats = new_feats

            feat = np.mean(list(node_feats.values()), axis=0)
            ddgm_features.append(feat)
    return np.array(ddgm_features)


def extract_chgm(subgraphs, d=16):
    """Extract CHGM high-order co-occurrence features via hypergraph spectral embedding."""
    chgm_features = []
    for period in sorted(subgraphs.keys()):
        for g in subgraphs[period]:
            hyperedges = []
            for v in g.nodes():
                in_edges = g.in_edges(v, data=True)
                times = [d['timestamp'] for _, _, d in in_edges]
                if not times:
                    continue
                latest = max(times)
                group = [u for u, _, d in in_edges if latest - d['timestamp'] <= 30]
                if len(group) > 1:
                    hyperedges.append(set(group))

            if not hyperedges:
                chgm_features.append(np.zeros(d))
                continue

            nodes = list(g.nodes())
            H = np.zeros((len(nodes), len(hyperedges)))
            for j, hedge in enumerate(hyperedges):
                for i, n in enumerate(nodes):
                    if n in hedge:
                        H[i, j] = 1

            Dv = np.diag(H @ H.T)
            De = np.diag(H.T @ H)
            Dv_inv = np.linalg.inv(Dv + np.eye(Dv.shape[0]) * 1e-5)
            De_inv = np.linalg.inv(De + np.eye(De.shape[0]) * 1e-5)
            W = np.diag([len(h) for h in hyperedges])
            L = np.eye(len(nodes)) - Dv_inv @ H @ W @ De_inv @ H.T

            eigval, eigvec = np.linalg.eigh(L)
            embedding = eigvec[:, 1:d+1]  # skip trivial 0 eigenvalue
            chgm_features.append(np.mean(embedding, axis=0))
    return np.array(chgm_features)


def analyze_mixed_order_features(target, subgraphs):
    """Main function to extract and save DDGM and CHGM features."""
    print(f"[INFO] Extracting mixed-order features for target: {target}")
    ddgm = extract_ddgm(subgraphs)
    chgm = extract_chgm(subgraphs)

    output_dir = os.path.join('data/processed', target)
    ensure_dir(output_dir)
    np.save(os.path.join(output_dir, 'ddgm.npy'), ddgm)
    np.save(os.path.join(output_dir, 'chgm.npy'), chgm)
    print(f"[INFO] DDGM and CHGM features saved to {output_dir}")