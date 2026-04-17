import numpy as np
from envs.network_factory import Network
from utils.operations import consensus_degree
from utils.visualization import plot_opinion_distribution, plot_network, plot_simulation_summary
import copy


def run_passive_simulation(num_steps=1, num_nodes=1000, hk_threshold=0.3):
    # Create a network and run the passive simulation
    network = Network(num_nodes=num_nodes, hk_threshold=hk_threshold)
    network_start = copy.deepcopy(network)

    print(f"Starting passive simulation for {num_steps} steps.")
    print(f"Initial Consensus: {consensus_degree(network.opinions):.4f}")
    plot_network(network, step=0, title="Initial Social Network", save_path="outputs/sf_initial_social_network.png")

    consensus_history = []
    opinions_history = []
    opinions_history.append(network.opinions.detach().cpu().numpy())
    
    for step in range(num_steps):
        network.apply_dynamics()  # Update opinions based on current weights
        cd = consensus_degree(network.opinions)
        consensus_history.append(cd)
        opinions_history.append(network.opinions.detach().cpu().numpy())

        if step % 10 == 0:
            print(f"Step {step}: Consensus Degree = {cd:.4f}")
        
    # Visualize ONLY the final state
    plot_network(network, step=num_steps, title="Final Social Network", save_path="outputs/sf_final_social_network.png")
    plot_simulation_summary(opinions_history, network_start, network, title="Passive H-K Simulation", save_path="outputs/sf_passive_simulation_summary.png")
    print(f"Final Consensus: {consensus_degree(network.opinions):.4f}")

    print(f"Final Consensus: {consensus_history[-1]:.4f}")
    return consensus_history

if __name__ == "__main__":
    run_passive_simulation()