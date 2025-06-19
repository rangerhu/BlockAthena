import networkx as nx

def relabel_graph_nodes(graph, prefix='node'):
    """Relabel graph nodes with a prefix and index."""
    mapping = {node: f"{prefix}_{i}" for i, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping)