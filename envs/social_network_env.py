import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from envs.network_factory import create_scale_free_weighted_directed_network, apply_hk_dynamics

class SocialNetworkEnv(gym.Env):
    def __init__(self, num_nodes=50, max_steps=100, cd_threshold=0.95, hk_threshold=0.1, penalty_coeff=0.1, success_bonus=10, action_lower = -0.1, action_upper = 0.1):
        super(SocialNetworkEnv, self).__init__()

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.cd_threshold = cd_threshold
        self.hk_threshold = hk_threshold
        self.penalty_coeff = penalty_coeff
        self.success_bonus = success_bonus
        self.current_step = 0

        #Action Space: Adjustments to the weight matrix (N x N)
        self.action_space = spaces.Box(low=action_lower, high=action_upper, shape=(num_nodes * num_nodes,), dtype=np.float64)

        #Observation Space: opinions, weight matix and communities
        self.observation_space = spaces.Dict({
            'opinions': spaces.Box(low=-1.0, high=1.0, shape=(num_nodes,), dtype=np.float64),
            'weights': spaces.Box(low=0.0, high=1.0, shape=(num_nodes, num_nodes), dtype=np.float64),
            #TODO: update communities in the future
        })



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        #Create a scale-free weighted directed network
        self.G = create_scale_free_weighted_directed_network(self.num_nodes)

        #Initialize opinions randomly between -1 and 1
        self.opinions = np.random.uniform(-1, 1, size=(self.num_nodes,)).astype(np.float64)

        #Initialize weight matrix
        self.W = nx.to_numpy_array(self.G, weight='weight').astype(np.float64)

        observation = {
            'opinions': self.opinions,
            'weights': self.W,
        }
        info = {}
        return observation, info

    def step(self, action):
        # store previous consensus and weights to calculate reward
        prev_consensus = self.consensus_degree()
        prev_weights = self.W.copy()
        terminated = False

        #1. Apply action: reshape action back to N X N and add to W
        self.apply_action(action)

        #3. Update opinions using H-K model
        new_opinions, gated_weights = apply_hk_dynamics(self.opinions, self.W, self.hk_threshold)
        self.opinions = new_opinions

        #4. Calculate reward
        reward, terminated = self.calculate_reward(prev_consensus, prev_weights)

        #5. Check if max steps reached
        self.current_step += 1
        truncated = terminated or (self.current_step >= self.max_steps)

        #update observation
        observation = {
            'opinions': self.opinions.astype(np.float64),
            'weights': gated_weights.astype(np.float64), # Agents observe the gated weights after applying H-K dynamics
        }
        info = {}

        return observation, reward, terminated, truncated, info

    def apply_action(self, action):
        # Reshape action back to N x N and add to W
        action_matrix = action.reshape((self.num_nodes, self.num_nodes))

        # Create a mask for existing nodes (where there are edges in the original graph)
        existing_edges_mask = (self.W > 0).astype(float)

        # Apply action only to existing edges (including self-loops)
        self.W = self.W + (action_matrix * existing_edges_mask)

        # Ensure weights remain non-negative
        self.W = np.clip(self.W, a_min=0, a_max=None)

        #TODO: check if action will already make the weights row stochastic
        # Normalize rows of matrix
        row_sums = self.W.sum(axis=1, keepdims=True)
        self.W = np.divide(self.W, row_sums, out=np.zeros_like(self.W), where=row_sums!=0)

    def calculate_reward(self, prev_consensus, prev_weights):
        reward = 0

        # A: Reward for increasing consensus (decreasing variance)
        current_consensus = self.consensus_degree()
        delta_consensus = current_consensus - prev_consensus

        # B: Penalty for large changes in weights to encourage stability(Frobenious norm of weight change)
        change_effort = np.linalg.norm(self.W - prev_weights, 'fro')

        # C: Success bonus for achieving consensus (if variance is below a certain threshold)
        success_bonus = 0
        terminated = False
        if current_consensus > self.cd_threshold:  # Consensus degree is high enough to consider consensus achieved
            success_bonus = self.success_bonus
            terminated = True

        # Combine components into final reward
        reward = delta_consensus - self.penalty_coeff * change_effort + success_bonus

        return reward, terminated

    def consensus_degree(self):
        #Calculating Opinion Deviation from the mean (range 0 to 1, where 1 means perfect consensus)
        mean_opinion = np.mean(self.opinions)
        absolute_deviations = np.abs(self.opinions - mean_opinion)
        total_deviation = np.sum(absolute_deviations)
        consensus_degree = 1 - (total_deviation / self.num_nodes)  # Normalize by number of nodes to get a score between 0 and 1

        return consensus_degree