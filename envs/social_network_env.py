import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from envs.network_factory import create_scale_free_weighted_directed_network

class SocialNetworkEnv(gym.Env):
    def __init__(self, num_nodes=50, max_steps=100, gamma_threshold=0.01):
        super(SocialNetworkEnv, self).__init__()

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.gamma_threshold = gamma_threshold
        self.current_step = 0

        #Action Space: Adjustments to the weight matrix (N x N)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(num_nodes * num_nodes,), dtype=np.float32)

        #Observation Space: opinions, weight matix and communities
        self.observation_space = spaces.Dict({
            'opinions': spaces.Box(low=-1.0, high=1.0, shape=(num_nodes,), dtype=np.float32),
            'weights': spaces.Box(low=0.0, high=1.0, shape=(num_nodes, num_nodes), dtype=np.float32),
            'communities': spaces.Box(low=0, high=num_nodes, shape=(num_nodes,), dtype=np.int32)
        })



    def reset(self, seed=None, options=None):
    
    def step(self, action):