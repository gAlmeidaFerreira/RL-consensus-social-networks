import networkx as nx
import numpy as np
import torch
from utils.operations import normalize_weights

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
    
    def __init__(self, num_nodes=50, m=3, hk_threshold=0.1, opinion_dynamics=None, device=None):
        """
        Initialize the Network.
        
        Args:
            num_nodes: Number of nodes in the network (default: 50)
            m: Number of edges to attach from a new node to existing nodes (default: 3)
            hk_threshold: Threshold for Hegselmann-Krause dynamics (default: 0.1)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_nodes = num_nodes
        self.m = m
        self.hk_threshold = hk_threshold
        self.opinion_dynamics = opinion_dynamics
        self._step_count = 0
        # Initialize the graph and weights
        self._create_network()
        
        # Initialize opinions randomly between -1 and 1
        self.opinions = torch.rand(self.num_nodes, dtype=torch.float64, device=self.device) * 2 - 1  # Uniform(-1, 1)
        
        # Store current gated weights (for dynamics application)
        self._current_gated_weights = self.weights.clone() if isinstance(self.weights, torch.Tensor) else self.weights.copy()
    
    def _create_network(self):
        """
        Create a scale-free weighted directed network using the Barabási-Albert model.
        Sets self.graph and self.weights as instance attributes.
        """
        #TODO: #5 For larger networks consider using torch_geometric or sparse representations to handle memory efficiently.
        # Create a scale-free network using Barabási-Albert model
        G = nx.barabasi_albert_graph(self.num_nodes, m=self.m)
        
        # Convert to directed graph
        G = G.to_directed()

        for node in range(self.num_nodes):
            G.add_edge(node, node, weight=1.0)  # Add self-loops with weight 1.0

        # Build and normalize the weight matrix on GPU (fallback to CPU when CUDA is unavailable).
        device = self.device
        adjacency_tensor = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float64, device=device)
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long, device=device)
        adjacency_tensor[edge_index[:, 0], edge_index[:, 1]] = 1.0

        # Power(a=2.5) samples can be generated as U^(1/a), where U ~ Uniform(0, 1).
        random_weights = torch.rand((self.num_nodes, self.num_nodes), dtype=torch.float64, device=device).pow(1.0 / 2.5)
        weighted_matrix = random_weights * adjacency_tensor
        weighted_matrix.fill_diagonal_(1.0)

        normalized_matrix = normalize_weights(weighted_matrix, return_tensor=True)
        
        self.graph = G
        
        # Keep weights on device to avoid host copies during network creation.
        self.weights = normalized_matrix
        # Store original weights
        self.original_weights = self.weights.clone()
    
    def apply_dynamics(self, dynamics=None):
        """
        Apply Hegselmann-Krause dynamics to update opinions based on current weights.
        
        Returns:
            Tuple of (new_opinions, gated_weights) where:
                - new_opinions: Updated opinion vector
                - gated_weights: Gated weight matrix used for update
        """
        dynamics = dynamics or self.opinion_dynamics or apply_hk_dynamics

        new_opinions, gated_weights = dynamics(
            self.opinions, 
            self.weights, 
            self.hk_threshold
        )

        if not isinstance(new_opinions, torch.Tensor) or not isinstance(gated_weights, torch.Tensor):
            raise TypeError("dynamics must return (torch.Tensor, torch.Tensor)")
        
        # Update internal state
        self.opinions = new_opinions
        self._current_gated_weights = gated_weights
        self._step_count += 1
        
        return new_opinions, gated_weights
    
    def reset(self):
        """Reset the network to initial state with new random opinions."""
        self.opinions = torch.rand(self.num_nodes, dtype=torch.float64, device=self.device) * 2 - 1  # Uniform(-1, 1)
        self.weights = self.original_weights.clone() if isinstance(self.original_weights, torch.Tensor) else self.original_weights.copy()
        self._current_gated_weights = self.weights.clone() if isinstance(self.weights, torch.Tensor) else self.weights.copy()
        self._step_count = 0
    
    @property
    def step_count(self):
        """Get the current step count."""
        return self._step_count

def apply_hk_dynamics(opinions, weights, epsilon, return_tensor=True):
    """
    Apply Hegselmann-Krause dynamics to opinions based on weighted influence.
    
    Args:
        opinions: Opinion vector (num_nodes,)
        weights: Weight/adjacency matrix (num_nodes x num_nodes)
        epsilon: Confidence threshold for opinion updates
        return_tensor: Whether to return tensors or numpy arrays
        device: Device to perform computations on

    Returns:
        Tuple of (new_opinions, gated_weights) where:
            - new_opinions: Updated opinion vector
            - gated_weights: Gated weight matrix used for update
    """

    if not isinstance(opinions, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("apply_hk_dynamics expects torch.Tensor inputs for opinions and weights")

    opinions_tensor = opinions.flatten()
    weights_tensor = weights

    # Calculate pairwise opinion distances and build the confidence mask.
    dist_matrix = torch.abs(opinions_tensor.unsqueeze(1) - opinions_tensor.unsqueeze(0))
    influence_mask = (dist_matrix <= epsilon).to(torch.float64)

    # Normalize using shared utility to keep row-stochastic behavior consistent.
    gated_weights = normalize_weights(weights_tensor * influence_mask, return_tensor=True)

    # Update opinions using weighted neighborhood influence.
    new_opinions = torch.matmul(gated_weights, opinions_tensor)

    if return_tensor:
        return new_opinions, gated_weights

    return (
        new_opinions.detach().cpu().numpy().astype(np.float64),
        gated_weights.detach().cpu().numpy().astype(np.float64),
    )

