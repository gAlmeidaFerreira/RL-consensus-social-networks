from envs.network_factory import Network

# Create a network with specified parameters
network = Network(num_nodes=10, m=2)

# Access the underlying graph
G = network.graph
print(list(G.edges.data()))

# You can also access network state
print(f"Opinions: {network.opinions}")
print(f"Weights shape: {network.weights.shape}")