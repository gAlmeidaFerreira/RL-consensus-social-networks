import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from envs.network_factory import Network, apply_hk_dynamics
from utils.operations import consensus_degree, normalize_weights


class SocialNetworkEnv(gym.Env):
    def __init__(self, num_nodes=50, 
                 max_steps=500, 
                 cd_threshold=0.95, 
                 hk_threshold=0.2,
                 delta_consensus_coeff=50, 
                 change_penalty_coeff=0.1, 
                 success_bonus=10, 
                 action_lower=-0.2, 
                 action_upper=0.2,
                 topo_penalty_coeff=0.01, 
                 opinion_dynamics=None,
                 network=None):
        super(SocialNetworkEnv, self).__init__()

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.cd_threshold = cd_threshold
        self.hk_threshold = hk_threshold
        self.change_penalty_coeff = change_penalty_coeff
        self.success_bonus = success_bonus
        self.topo_penalty_coeff = topo_penalty_coeff
        self.delta_consensus_coeff = delta_consensus_coeff
        self.opinion_dynamics = opinion_dynamics
        self.current_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use provided network or create a new one
        if network is not None:
            self.network = network
            self.num_nodes = network.num_nodes
            if hasattr(network, "device"):
                self.device = network.device
        else:
            self.network = Network(
                num_nodes=num_nodes,
                hk_threshold=hk_threshold,
                opinion_dynamics=opinion_dynamics,
                device=self.device,
            )

        # Action Space: Adjustments to the weight matrix (N x N)
        self.action_space = spaces.Box(low=action_lower, high=action_upper, shape=(self.num_nodes * self.num_nodes,), dtype=np.float64)

        # Observation Space: opinions, weight matrix and communities
        self.observation_space = spaces.Dict({
            'opinions': spaces.Box(low=-1.0, high=1.0, shape=(self.num_nodes,), dtype=np.float64),
            'weights': spaces.Box(low=0.0, high=1.0, shape=(self.num_nodes, self.num_nodes), dtype=np.float64),
            # TODO: #4 update communities in the future
        })

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float64)
        return np.asarray(value, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Reset network opinions and state
        self.network.reset()

        # Keep core state on device for matrix-heavy operations.
        self.network.opinions = self.network.opinions.flatten()

        # Calculate initial degree variance
        adj = (self.network.weights > 1e-4).to(torch.float64)
        initial_degrees = torch.sum(adj, dim=1)  # Out-degrees
        self.initial_degree_variance = torch.var(initial_degrees, unbiased=False)

        # Build observation from network state
        observation = {
            'opinions': self._to_numpy(self.network.opinions),
            'weights': self._to_numpy(self.network.weights),
        }
        info = {}
        return observation, info  

    def step(self, action):
        with torch.no_grad():
            # store previous consensus and weights to calculate reward
            prev_consensus = consensus_degree(self.network.opinions)
            prev_weights = self.network.weights.clone()
            terminated = False

            # 1. Apply action: reshape action back to N X N and add to W
            self.apply_action(action)

            # 2. Update opinions using H-K model via network dynamics
            dynamics_fn = self.opinion_dynamics
            if dynamics_fn is None:
                dynamics_fn = lambda opinions, weights, epsilon: apply_hk_dynamics(
                    opinions, weights, epsilon, return_tensor=True
                )

            new_opinions, gated_weights = self.network.apply_dynamics(dynamics_fn)
            if not isinstance(new_opinions, torch.Tensor) or not isinstance(gated_weights, torch.Tensor):
                raise TypeError("Dynamics function must return torch.Tensor outputs")
            self.network.opinions = new_opinions.flatten()
            self.network.weights = gated_weights

            # 3. Calculate reward
            reward, terminated, current_consensus, change_effort, topo_deviation = self.calculate_reward(prev_consensus, prev_weights)

            # 4. Check if max steps reached
            self.current_step += 1
            truncated = terminated or (self.current_step >= self.max_steps)

        # update observation
        observation = {
            'opinions': self._to_numpy(self.network.opinions),
            'weights': self._to_numpy(gated_weights),  # Agents observe the gated weights after applying H-K dynamics
        }

        info = {
            "consensus_degree": current_consensus,
            "change_effort": change_effort,
            "topo_deviation": topo_deviation,
            "reward_value": reward,
        }

        return observation, float(reward), terminated, truncated, info

    def apply_action(self, action):
        weights_tensor = self.network.weights

        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).to(device=self.device, dtype=weights_tensor.dtype)
        elif isinstance(action, torch.Tensor):
            action_tensor = action.to(device=self.device, dtype=weights_tensor.dtype)
        else:
            raise TypeError("apply_action expects action as torch.Tensor or numpy.ndarray")

        action_tensor = action_tensor.view(self.num_nodes, self.num_nodes)

        # Apply action updates and keep weights non-negative.
        updated_weights = torch.clamp(weights_tensor + action_tensor, min=0.0)

        # TODO: #3 check if action will already make the weights row stochastic and if not, normalize them
        normalized_weights = normalize_weights(updated_weights, return_tensor=True)

        self.network.weights = normalized_weights

    def calculate_reward(self, prev_consensus, prev_weights):
        # A: Reward for increasing consensus (decreasing variance)
        current_consensus = consensus_degree(self.network.opinions)
        delta_consensus = current_consensus - prev_consensus

        # B: Penalty for large changes in weights to encourage stability (Frobenius norm of weight change)
        current_weights_tensor = self.network.weights
        prev_weights_tensor = prev_weights
        change_effort = torch.linalg.norm(current_weights_tensor - prev_weights_tensor, ord="fro")

        # C: Topology change penalty (squared difference for existing edges to discourage drastic topology changes)
        current_adj = (current_weights_tensor > 1e-4).to(torch.float64)
        current_degrees = torch.sum(current_adj, dim=1)  # Out-degrees
        current_degree_variance = torch.var(current_degrees, unbiased=False)
        topo_deviation = (current_degree_variance - self.initial_degree_variance) ** 2

        # D: Success bonus for achieving consensus (if variance is below a certain threshold)
        success_bonus = 0
        terminated = False
        if current_consensus > self.cd_threshold:  # Consensus degree is high enough to consider consensus achieved
            success_bonus = self.success_bonus
            terminated = True

        # Combine components into final reward
        reward_tensor = (
            self.delta_consensus_coeff * delta_consensus
            - self.change_penalty_coeff * change_effort
            - self.topo_penalty_coeff * topo_deviation
            + success_bonus
        )

        reward = reward_tensor.item()
        change_effort = change_effort.item()
        topo_deviation = topo_deviation.item()
        current_consensus = current_consensus.item()

        return reward, terminated, current_consensus, change_effort, topo_deviation