import networkx as nx
import numpy as np

class Network:
    """
    A stateful network object that manages graph structure, weights, and opinion dynamics.
    
    Attributes:
        num_nodes (int): Number of nodes in the network
        hk_threshold (float): Threshold for Hegselmann-Krause dynamics
        graph (nx.DiGraph): NetworkX directed graph
        weights (np.ndarray): Weight matrix (num_nodes x num_nodes)
        opinions (np.ndarray): Opinion vector (num_nodes,)
    """
    
    def __init__(self, num_nodes=50, m=3, hk_threshold=0.1):
        """
        Initialize the Network.
        
        Args:
            num_nodes: Number of nodes in the network (default: 50)
            m: Number of edges to attach from a new node to existing nodes (default: 3)
            hk_threshold: Threshold for Hegselmann-Krause dynamics (default: 0.1)
        """
        self.num_nodes = num_nodes
        self.m = m
        self.hk_threshold = hk_threshold
        self._step_count = 0
        
        # Initialize the graph and weights
        self._create_network()
        
        # Initialize opinions randomly between -1 and 1
        self.opinions = np.random.uniform(-1, 1, size=(self.num_nodes,)).astype(np.float64)
        
        # Store current gated weights (for dynamics application)
        self._current_gated_weights = self.weights.copy()
    
    def _create_network(self):
        """
        Create a scale-free weighted directed network using the Barabási-Albert model.
        Sets self.graph and self.weights as instance attributes.
        """
        # Create a scale-free network using Barabási-Albert model
        G = nx.barabasi_albert_graph(self.num_nodes, m=self.m)
        
        # Convert to directed graph
        G = G.to_directed()

        for node in range(self.num_nodes):
            G.add_edge(node, node, weight=1.0)  # Add self-loops with weight 1.0    
        
        # Adding self-loops to ensure that each node can influence itself
        for u, v in G.edges():
            if u != v:
                # Assign random weights to edges (using a power-law distribution)
                G[u][v]['weight'] = np.random.power(a=2.5)
        
        # Normalize weights to make them row stochastic
        for node in G.nodes():
            out_edges = list(G.out_edges(node, data=True))
            total = sum(d['weight'] for _, _, d in out_edges)
            if total > 0:
                for _, v, d in out_edges:
                    d['weight'] /= total
        
        self.graph = G
        
        # Extract weight matrix from graph
        self.weights = nx.to_numpy_array(self.graph, weight='weight').astype(np.float64)
        # Store original weights
        self.original_weights = self.weights.copy()
    
    def apply_dynamics(self):
        """
        Apply Hegselmann-Krause dynamics to update opinions based on current weights.
        
        Returns:
            Tuple of (new_opinions, gated_weights) where:
                - new_opinions: Updated opinion vector
                - gated_weights: Gated weight matrix used for update
        """
        new_opinions, gated_weights = apply_hk_dynamics(
            self.opinions, 
            self.weights, 
            self.hk_threshold
        )
        
        # Update internal state
        self.opinions = new_opinions
        self._current_gated_weights = gated_weights
        self._step_count += 1
        
        return new_opinions, gated_weights
    
    def reset(self):
        """Reset the network to initial state with new random opinions."""
        self.opinions = np.random.uniform(-1, 1, size=(self.num_nodes,)).astype(np.float64)
        self.weights = self.original_weights.copy()
        self._current_gated_weights = self.weights.copy()
        self._step_count = 0
    
    @property
    def step_count(self):
        """Get the current step count."""
        return self._step_count

def apply_hk_dynamics(opinions, weights, epsilon):
    """
    Apply Hegselmann-Krause dynamics to opinions based on weighted influence.
    
    Args:
        opinions: Opinion vector (num_nodes,)
        weights: Weight/adjacency matrix (num_nodes x num_nodes)
        epsilon: Confidence threshold for opinion updates
    
    Returns:
        Tuple of (new_opinions, gated_weights) where:
            - new_opinions: Updated opinion vector
            - gated_weights: Gated weight matrix used for update
    """
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

    # if node is isolated, it keeps its opinion
    isolated_mask = (row_sums.flatten() == 0)
    gated_weights[isolated_mask, isolated_mask] = 1.0

    # Update opinions based on the weighted average of neighbors' opinions
    new_opinions = np.dot(gated_weights, opinions)
    return new_opinions, gated_weights

