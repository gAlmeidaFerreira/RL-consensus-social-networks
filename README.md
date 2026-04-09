# RL-consensus-social-networks
The goal of the project is to build a Reinforcement learning model that can prevent polarization in a social network

## File Structure
polarization_rl/
├── envs/
│   ├── init.py
│   ├── social_network_env.py   # The Gymnasium environment class
│   └── network_factory.py      # Scale-free graph generation logic and Opinion Dynamics
├── agents/
│   ├── init.py
│   ├── common/
│   │   ├── networks.py        # Shared GNN and MLP architectures
│   │   └── replay_buffer.py    # Experience replay for SAC/MAAC
│   ├── sac_agent.py            # Single-agent Soft Actor-Critic logic
│   └── maac_agent.py           # Multi-agent Actor-Critic logic
├── utils/
│   ├── community_detection.py  # Wrapper for the two detection algorithms
│   ├── metrics.py              # Polarization & consensus degree calcs
│   └── visualization.py       # Graph plotting and opinion distributions
├── config/
│   └── hyperparams.yaml        # Learning rates, gamma, epsilon, etc.
├── train.py                   # Entry point for training models
├── evaluate.py                # Script to compare the two algorithms
└── requirements.txt

## V1
environment:

- Network topology will be scale free network with directed weighted edges (weight matrix is line stochastic) and initially 50 nodes
- each node will have a continuous value opinion ranging from -1 to 1
- the opinion change: select the neighbors by confidence bound (HK model) then: weighted mean of the neighbors

RL algorithm:

- community detection: I want to make a community detection based on connections and opinion values
- Use two different algorithms and compare the two performances at the end
    - Soft Actor Critic - Critic calculates the q-value based on long term reward, actor will estimate the best policy and use a entropy regularization to avoid collapsing
    - MARL actor critic: one critic calculating the q value based on the long term reward and 1 actor per community, each actor will be able to act only over the node inside its community, calculating the best policy with entropy regularization
- state space: the graph with nodes, edges and the community of each node
- Action space: the RL agents will be able to change the connections between nodes, increasing or lowering the weight of connections
    - For MARL the actors will be able to act only over the connections between nodes inside its community
- Reward:
    - + global consensus degree increase calculated at each time step
    - - (penalty) connections between nodes from different communities, and follow the logic
        - for SAC - lower penalty for increasing connection same community nodes, high penalty to increasing connection different communities
        - For MAAC - the penalty is inverse proportional to the difference in opinion of the nodes
    - + Reaching global consensus degree within threshold  = gamma. also means termination
- There will be a maximum number of steps representing the maximum time for termination.

