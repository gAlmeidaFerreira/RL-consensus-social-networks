import os
from typing import Callable
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.social_network_env import SocialNetworkEnv
from agents.common.networks import SocialNetworkFeatureExtractor
from agents.common.callbacks import PolarizationMetricsCallback
from utils.visualization import plot_opinion_distribution, plot_network

def learning_rate_schedule(initial_lr: float) -> Callable[[float], float]:
    """
    Creates a learning rate schedule function that decays the learning rate over time.
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return schedule

def make_social_env(num_nodes=50):
    def _init():
        return SocialNetworkEnv(num_nodes=num_nodes)
    return _init

class SACAgent:
    def __init__(self, env=None, log_dir="logs/sac", initial_lr=3e-4, num_nodes=50, n_envs=8):
        # Logs setup
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # VecEnv setup for parallel environments
        if n_envs > 1:
            env_fns = [make_social_env(num_nodes) for _ in range(n_envs)]
            self.env = SubprocVecEnv(env_fns)
        else:
            self.env = make_social_env(num_nodes)()

        # Policy configuration
        self.policy_kwargs = dict(
            features_extractor_class=SocialNetworkFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(
                pi=[512, 512, 256],  # Actor (Policy)
                qf=[512, 512, 256]   # Critic (Q-function)
            ),
            activation_fn=torch.nn.ReLU
        )

        # SAC model initialization
        self.model = SAC(
            policy='MultiInputPolicy',
            env=self.env,
            learning_rate=learning_rate_schedule(initial_lr), # LR schedule
            buffer_size=1_000_000,
            learning_starts=5000,                             # Allow some random exploration before learning
            batch_size=512,                                  # Larger batch size for more stable updates
            tau=0.005,                                        # Soft update coefficient
            gamma=0.99,                                       # Discount factor
            train_freq=(1, 'step'),                           # Train every step
            gradient_steps=10,                                  # Number of gradient steps per training iteration
            ent_coef='auto',                                      # Automatically adjust entropy coefficient
            target_entropy='auto',                                    # Automatically set target entropy
            policy_kwargs=self.policy_kwargs,
            tensorboard_log=self.log_dir,
            verbose=1,
            device='cuda'
        )
        print(f"SAC Model initialized on device: {self.model.device}")

    def train(self, total_timesteps=500_000, callback=PolarizationMetricsCallback()):
        """Execute the training loop."""
        print(f"Starting training for {total_timesteps} timesteps...")
        if hasattr(self.env, 'network'):
            plot_network(
                self.env.network,
                title="Initial Social Network ( SAC Agent Start )",
                save_path="outputs/initial_network_visualization.png" # Saves the file
                # If you don't provide save_path, it will call plt.show()
        )
        else:
            print("Error: Could not find network object within environment to visualize.")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="sac_run",
            reset_num_timesteps=True
        )

    def save(self, path="sac_agent"):
        """Save the trained model."""
        self.model.save(path)

    def load(self, path="sac_agent"):
        """Load a pre-trained model."""
        self.model = SAC.load(path, env=self.env)

    def predict(self, observation, deterministic=True):
        """Predict an action given an observation."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

if __name__ == "__main__":
    agent = SACAgent()
    agent.train(total_timesteps=500_000)
    agent.save("sac_agent_final")

