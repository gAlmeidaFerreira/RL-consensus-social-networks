import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

def plot_opinion_distribution(opinions, step, title="Opinion Distribution"):
    """
    Plots a histogram of opinions to visualize polarization clusters.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(opinions, bins=20, range=(-1, 1), color='skyblue', edgecolor='black')
    if step is not None:
        plt.title(f"{title} - Step {step}")
    else:
        plt.title(title)
    plt.xlabel("Opinion Value")
    plt.ylabel("Number of Agents")
    plt.xlim(-1.1, 1.1)
    plt.grid(axis='y', alpha=0.5)
    plt.show() # If running in a notebook, this renders immediately

def plot_network(network, step=None, title="Social Network", save_path=None, labels=False):
    """
    Generates a 2D visualization of the network nodes, opinions, and weights.

    Args:
        network: An instance of your Network class (from envs.network_factory).
                 Must have 'opinions' (N, 1 tensor), 'weights' (N, N tensor).
        title (str): Title of the plot.
        save_path (str, optional): If provided, saves the image to this path.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(network.opinions, torch.Tensor):
        opinions = network.opinions.cpu().numpy().flatten()
    else:
        opinions = network.opinions

    if isinstance(network.weights, torch.Tensor):
        weights = network.weights.cpu().numpy()
    else:
        weights = network.weights
    num_nodes = opinions.shape[0]

    if num_nodes == 0:
        print("No nodes to plot.")
        return
    
    # Create a graph from the weight matrix
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i, opinion=opinions[i])
   
    edges_threshold = 1e-3  # Only plot edges with weight above this threshold
    edges_to_add = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if weights[i, j] > edges_threshold:
                edges_to_add.append((i, j, weights[i, j]))

    if edges_to_add:
        G.add_weighted_edges_from(edges_to_add)

    # Visualization Setup
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    if step is not None:
        ax.set_title(f"{title} - Step {step}")
    else:
        ax.set_title(title)

    # Place nodes with same opinions closer together using a spring layout
    pos = {i: np.array([opinions[i] * 5.0, np.random.normal(0, 0.5)]) for i in range(num_nodes)}


    # Draw nodes with colors based on opinions
    cmap = plt.cm.get_cmap('bwr') # Blue-White-Red colormap
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    node_colors = [cmap(norm(G.nodes[n]['opinion'])) for n in G.nodes]

    nx.draw_networkx_nodes(G, pos=pos,
                           node_size=600,  
                           node_color=node_colors, 
                           edgecolors='black',
                           linewidths=1.0,
                           ax=ax)
    
    # Draw node labels
    if labels:
        nx.draw_networkx_labels(G, pos=pos, font_size=8, font_color='black', ax=ax)

    # Draw edges with widths based on weights
    if edges_to_add:
        all_weights = [d['weight'] for u, v, d in G.edges(data=True)]
        if all_weights:
            max_weight = max(all_weights)
            min_weight = min(all_weights)
            
            # normalize edge opacities based on weights
            if max_weight > min_weight:
                alpha_weights = [(w - min_weight) / (max_weight - min_weight)*0.7 for w in all_weights]
            else:
                alpha_weights = [0.5] * len(all_weights)
        else:
            alpha_weights = []

        # Draw edges with varying widths and alpha
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            nx.draw_networkx_edges(G, pos=pos, 
                                   edgelist=[(u, v)],
                                   width=1.0,
                                   alpha=alpha_weights[i],
                                   edge_color='gray',
                                   arrows=True,
                                   arrowsize=10,
                                   ax=ax)
    
    # Add Legend for Opinions
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', label='Agent Opinion Degree')

    ax.axis('off')  # Hide axes

    # Save or show the plot
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Network visualization saved to {save_path}")
    else:
        print("Displaying network visualization.")
        plt.show()
    
    plt.close()

def plot_simulation_summary(opinion_history, network_start, network_end, title="Opinion Dynamics Summary", save_path=None):
    """
    Layout:
    [ Time Series (Left Column) | Initial State (Top Right) ]
    [ (Spanned)                | Final State   (Bottom Right) ]
    """
    fig = plt.figure(figsize=(16, 8))
    # Create a 2x2 grid where the first column (index 0) spans both rows
    grid = plt.GridSpec(2, 2, width_ratios=[2, 1], wspace=0.2, hspace=0.3)
    
    # --- LEFT COLUMN: Time Series (Spans both rows) ---
    ax_time = fig.add_subplot(grid[:, 0]) 
    history = np.array(opinion_history)
    for i in range(history.shape[1]):
        ax_time.plot(history[:, i], color='black', alpha=0.2, linewidth=0.8)
    
    ax_time.set_title(f"{title} - Opinion Evolution", fontsize=14)
    ax_time.set_xlabel("Time (steps)")
    ax_time.set_ylabel("Opinion Value")
    ax_time.grid(True, alpha=0.3)
    
    # --- RIGHT COLUMN: Initial State (Top) ---
    ax_start = fig.add_subplot(grid[0, 1])
    _draw_network_to_axis(ax_start, network_start, "Initial State")
    
    # --- RIGHT COLUMN: Final State (Bottom) ---
    ax_end = fig.add_subplot(grid[1, 1])
    _draw_network_to_axis(ax_end, network_end, "Final State")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Simulation summary saved to {save_path}")
    else:
        plt.show()

def _draw_network_to_axis(ax, network, title):
    """Helper to draw the polar rounded network on a specific matplotlib axis."""
    opinions = network.opinions.detach().cpu().numpy().flatten()
    num_nodes = len(opinions)
    
    # Create Graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Layout: Polar Rounded (Opinions mapped to angle)
    pos = {}
    angles = (opinions + 1.0) / 2.0 * 2 * np.pi
    for i in range(num_nodes):
        r = 1.0 + (np.random.rand() * 0.2)
        pos[i] = np.array([r * np.cos(angles[i]), r * np.sin(angles[i])])
    
    # Colors
    cmap = plt.cm.get_cmap('bwr')
    node_colors = [cmap((o + 1) / 2) for o in opinions]
    
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, 
                           edgecolors='black', linewidths=0.5, ax=ax)
    
    # Draw edges only for significant weights
    weights = network.weights.detach().cpu().numpy()
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if weights[i, j] > 0.05: # Threshold for visual clarity
                edges.append((i, j))
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, alpha=0.1, edge_color='grey', ax=ax)
    
    ax.set_title(title)
    ax.axis('off')