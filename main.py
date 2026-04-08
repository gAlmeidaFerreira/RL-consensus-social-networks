from envs import create_scale_free_weighted_directed_network

G = create_scale_free_weighted_directed_network(num_nodes=10, m=2)
print(list(G.edges.data()))