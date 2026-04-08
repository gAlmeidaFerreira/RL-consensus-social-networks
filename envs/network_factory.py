import networkx as nx
import numpy as np

def create_scale_free_weighted_directed_network(num_nodes=50):
    """
    Create a scale-free weighted directed network using the Barabási-Albert model.
    
    Args:
        num_nodes: Number of nodes in the network (default: 50)
    
    Returns:
        A directed NetworkX graph with row stochastic weighted edges
    """
    # Create a scale-free network using Barabási-Albert model
    G = nx.barabasi_albert_graph(num_nodes, attachment=2)
    
    # Convert to directed graph
    G = G.to_directed()
    
    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = nx.utils.random_sequence.powerlaw_sequence(1, exponent=2.5)[0]
    
    # Normalize weights to make them row stochastic
    for node in G.nodes():
        out_edges = G.out_edges(node, data=True)
        total_weight = sum(data['weight'] for _, _, data in out_edges)
        if total_weight > 0:
            for _, v, data in out_edges:
                data['weight'] /= total_weight
    
    return G