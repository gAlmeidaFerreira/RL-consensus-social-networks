import networkx as nx
import numpy as np

def create_scale_free_weighted_directed_network(num_nodes=50, m=2):
    """
    Create a scale-free weighted directed network using the Barabási-Albert model.
    
    Args:
        num_nodes: Number of nodes in the network (default: 50)
        m: Number of edges to attach from a new node to existing nodes.
    
    Returns:
        A directed NetworkX graph with row stochastic weighted edges
    """
    # Create a scale-free network using Barabási-Albert model
    G = nx.barabasi_albert_graph(num_nodes, m=m)
    
    # Convert to directed graph
    G = G.to_directed()

    # Adding self-loops to ensure that each node can influence itself
    G.add_edges_from([(i, i) for i in G.nodes()])
    
    # Add random weights to edges following a power-law distribution
    for u, v in G.edges():
        if u != v: # Don't overwrite the self-loop weight(TODO: can be changed to different stubbornness level in the future)
            # Assign weights from a power-law distribution (e.g., exponent=2.5)
            G[u][v]['weight'] = nx.utils.random_sequence.powerlaw_sequence(1, exponent=2.5)[0]
    
    # Normalize weights to make them row stochastic
    for node in G.nodes():
        out_edges = G.out_edges(node, data=True)
        total_weight = sum(data['weight'] for _, _, data in out_edges)
        if total_weight > 0:
            for _, v, data in out_edges:
                data['weight'] /= total_weight
    
    return G

def apply_hk_dynamics(opinions, weights, epsilon):
    # Calculate the distance matrix between opinions
    dist_matrix = np.abs(opinions[:, np.newaxis] - opinions[np.newaxis, :])

    # Select only those weights where the distance is less than or equal to epsilon
    influence_mask = (dist_matrix <= epsilon).astype(float)
    gated_weights = weights * influence_mask

    # normalize gated weights to ensure they sum to 1 for each node
    row_sums = gated_weights.sum(axis=1, keepdims=True)
    gated_weights = np.divide(gated_weights, row_sums, 
                              out=np.zeros_like(gated_weights), 
                              where=row_sums != 0)

    #if node is isolated, it keeps its opinion
    isolated_mask = (row_sums.flatten() == 0)
    gated_weights[isolated_mask, isolated_mask] = 1.0

    # Update opinions based on the weighted average of neighbors' opinions
    new_opinions = np.dot(gated_weights, opinions)
    return new_opinions, gated_weights

