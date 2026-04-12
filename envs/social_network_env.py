import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.network_factory import Network
from utils.metrics import consensus_degree


class SocialNetworkEnv(gym.Env):
    def __init__(self, num_nodes=50, 
                 max_steps=100, 
                 cd_threshold=0.95, 
                 hk_threshold=0.1, 
                 penalty_coeff=0.1, 
                 success_bonus=10, 
                 action_lower=-0.1, 
                 action_upper=0.1,
                 topo_penalty_coeff=0.1, 
                 network=None):
        super(SocialNetworkEnv, self).__init__()

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.cd_threshold = cd_threshold
        self.hk_threshold = hk_threshold
        self.penalty_coeff = penalty_coeff
        self.success_bonus = success_bonus
        self.topo_penalty_coeff = topo_penalty_coeff
        self.current_step = 0

        # Use provided network or create a new one
        if network is not None:
            self.network = network
            self.num_nodes = network.num_nodes
        else:
            self.network = Network(num_nodes=num_nodes, hk_threshold=hk_threshold)

        # Action Space: Adjustments to the weight matrix (N x N)
        self.action_space = spaces.Box(low=action_lower, high=action_upper, shape=(self.num_nodes * self.num_nodes,), dtype=np.float64)

        # Observation Space: opinions, weight matrix and communities
        self.observation_space = spaces.Dict({
            'opinions': spaces.Box(low=-1.0, high=1.0, shape=(self.num_nodes,), dtype=np.float64),
            'weights': spaces.Box(low=0.0, high=1.0, shape=(self.num_nodes, self.num_nodes), dtype=np.float64),
            # TODO: update communities in the future
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Reset network opinions and state
        self.network.reset()

        # Calculate initial degree variance
        adj = (self.network.weights > 1e-4).astype(float)
        initial_degrees = np.sum(adj, axis=1) # Out-degrees
        self.initial_degree_variance = np.var(initial_degrees)

        # Build observation from network state
        observation = {
            'opinions': self.network.opinions.copy().astype(np.float64),
            'weights': self.network.weights.copy().astype(np.float64),
        }
        info = {}
        return observation, info

    def step(self, action):
        # store previous consensus and weights to calculate reward
        prev_consensus = self.consensus_degree()
        prev_weights = self.network.weights.copy()
        terminated = False

        # 1. Apply action: reshape action back to N X N and add to W
        self.apply_action(action)

        # 2. Update opinions using H-K model via network dynamics
        new_opinions, gated_weights = self.network.apply_dynamics()

        # 3. Calculate reward
        reward, terminated, current_consensus, change_effort, topo_deviation = self.calculate_reward(prev_consensus, prev_weights)

        # 4. Check if max steps reached
        self.current_step += 1
        truncated = terminated or (self.current_step >= self.max_steps)

        # update observation
        observation = {
            'opinions': self.network.opinions.astype(np.float64),
            'weights': gated_weights.astype(np.float64),  # Agents observe the gated weights after applying H-K dynamics
        }

        info = {
            "consensus_degree": self.consensus_degree(),
            "change_effort": change_effort,
            "topo_deviation": topo_deviation
        }

        return observation, float(reward), terminated, truncated, info

    def apply_action(self, action):
        # Reshape action back to N x N and add to W
        action_matrix = action.reshape((self.num_nodes, self.num_nodes))

        # Apply action only to existing edges (including self-loops)
        self.network.weights = self.network.weights + action_matrix

        # Ensure weights remain non-negative
        self.network.weights = np.clip(self.network.weights, a_min=0, a_max=None)

        # TODO: check if action will already make the weights row stochastic
        # Normalize rows of matrix
        row_sums = self.network.weights.sum(axis=1, keepdims=True)
        self.network.weights = np.divide(
            self.network.weights, 
            row_sums, 
            out=np.zeros_like(self.network.weights), 
            where=row_sums != 0
        )

        isolated_mask = (row_sums.flatten() == 0)
        self.network.weights[isolated_mask, isolated_mask] = 1.0  # Ensure isolated nodes have self-loop

    def calculate_reward(self, prev_consensus, prev_weights):
        reward = 0

        # A: Reward for increasing consensus (decreasing variance)
        current_consensus = self.consensus_degree()
        delta_consensus = current_consensus - prev_consensus

        # B: Penalty for large changes in weights to encourage stability(Frobenious norm of weight change)
        change_effort = np.linalg.norm(self.network.weights - prev_weights, 'fro')

        # C: Topology change penalty (squared difference for existing edges to discourage drastic topology changes)
        current_adj = (self.network.weights > 1e-4).astype(float)
        current_degrees = np.sum(current_adj, axis=1) # Out-degrees
        current_degree_variance = np.var(current_degrees)
        topo_deviation = (current_degree_variance - self.initial_degree_variance) ** 2 # deviation from initial degree variance as a measure of topology change


        # C: Success bonus for achieving consensus (if variance is below a certain threshold)
        success_bonus = 0
        terminated = False
        if current_consensus > self.cd_threshold:  # Consensus degree is high enough to consider consensus achieved
            success_bonus = self.success_bonus
            terminated = True

        # Combine components into final reward
        reward = delta_consensus - self.penalty_coeff * change_effort - self.topo_penalty_coeff * topo_deviation + success_bonus

        return reward, terminated, current_consensus, change_effort, topo_deviation

    def consensus_degree(self):
        return consensus_degree(self.network.opinions)