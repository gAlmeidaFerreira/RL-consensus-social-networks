import numpy as np
from envs.network_factory import Network
from utils.operations import consensus_degree
from utils.visualization import plot_opinion_distribution


def run_passive_simulation(num_steps=100, num_nodes=50, hk_threshold=0.2):
    # Create a network and run the passive simulation
    network = Network(num_nodes=num_nodes, hk_threshold=hk_threshold)

    print(f"Starting passive simulation for {num_steps} steps.")
    print(f"Initial Consensus: {consensus_degree(network.opinions):.4f}")

    consensus_history = []
    
    for step in range(num_steps):
        network.apply_dynamics()  # Update opinions based on current weights
        cd = consensus_degree(network.opinions)
        consensus_history.append(cd)

        if step % 10 == 0:
            print(f"Step {step}: Consensus Degree = {cd:.4f}")
        
    # Visualize ONLY the final state
    plot_opinion_distribution(network.opinions, step=num_steps, title="Final Opinion Distribution")
    print(f"Final Consensus: {consensus_degree(network.opinions):.4f}")

    print(f"Final Consensus: {consensus_history[-1]:.4f}")
    return consensus_history

if __name__ == "__main__":
    run_passive_simulation()