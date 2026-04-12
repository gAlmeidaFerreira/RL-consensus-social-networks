import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class SocialNetworkFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that flattens the weight matrix and concatenates 
    it with the opinion vector.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):

        num_nodes = observation_space['opinions'].shape[0]
        input_size = num_nodes + num_nodes * num_nodes  # opinions + flattened weights

        super().__init__(observation_space, features_dim=features_dim)

        self.extract = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract and flatten components
        opinions = observations['opinions']
        weights_flat = torch.flatten(observations['weights'], start_dim=1)  # Flatten the weight matrix

        # Create one long state vector
        features = torch.cat([opinions, weights_flat], dim=1)

        return self.extract(features)

class MLPNetwork(nn.Module):
    """
    A standard 3-layer MLP backbone used by the Actor and Critic.
    In SAC, the Actor will use this to output Mean/Std, 
    and the Critic will use it to output a Q-value.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: int = 512):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)