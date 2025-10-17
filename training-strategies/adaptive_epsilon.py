"""
Adaptive Epsilon DQN Implementation
====================================
DQN with adaptive epsilon-greedy exploration:
- Epsilon adjusts based on recent reward performance
- High rewards -> lower epsilon (more exploitation)
- Low rewards -> higher epsilon (more exploration)
- Uniform random replay buffer (no prioritization)
- Target network (periodic updates)
- Training across multiple pole lengths (for generalization)

This implements reward-adaptive exploration strategy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque

from test_script import QNetwork
from utils.utils import save_training_results, plot_training_curve, compute_training_stats


class UniformReplayBuffer:
    """Standard uniform random replay buffer (no prioritization)."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Store a transition."""
        # Clone tensors to avoid in-place modification issues
        s = state.clone() if isinstance(state, torch.Tensor) else state
        ns = next_state.clone() if isinstance(next_state, torch.Tensor) else next_state
        self.buffer.append((s, action, reward, ns, done))
    
    def sample(self, batch_size):
        """Sample a batch uniformly at random."""
        samples = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


def train_dqn_step(q_network, target_network, replay_buffer, optimizer, batch_size, gamma):
    """
    Perform one training step using standard DQN with uniform replay.
    
    Args:
        q_network: Main Q-network
        target_network: Target Q-network
        replay_buffer: UniformReplayBuffer
        optimizer: Optimizer
        batch_size: Batch size
        gamma: Discount factor
    """
    # Sample batch uniformly
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # Compute current Q-values
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q-values
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        not_done = (~dones).to(dtype=torch.float32)
        target_q_values = rewards + (gamma * next_q_values * not_done)
    
    # Compute loss (standard MSE)
    loss = nn.MSELoss()(current_q_values, target_q_values)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_baseline_dqn(learning_rate=0.01, gamma=0.99, episodes=500, 
                       buffer_capacity=10000, batch_size=32,
                       epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                       target_update_freq=10, log_freq=25):
    """
    Train a baseline DQN with uniform replay and multi-pole training.
    
    Args:
        learning_rate: Learning rate for Adam optimizer
        gamma: Discount factor
        episodes: Number of training episodes
        buffer_capacity: Replay buffer capacity
        batch_size: Training batch size
        epsilon_start: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Epsilon decay per episode
        target_update_freq: Update target network every N episodes
        log_freq: Print progress every N episodes
    
    Returns:
        (plot_avg_rewards, q_network.state_dict(), episode_rewards)
    """
    print("="*60)
    print("ADAPTIVE EPSILON DQN TRAINING")
    print("="*60)
    print("Configuration:")
    print(f"  Method: DQN with Adaptive Epsilon")
    print(f"  Episodes: {episodes}")
    print(f"  Buffer Capacity: {buffer_capacity}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_min} (decay: {epsilon_decay})")
    print("="*60 + "\n")
    
    # Initialize environment and networks
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 2000
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    q_network = QNetwork(state_dim=state_dim, action_dim=action_dim)
    target_network = QNetwork(state_dim=state_dim, action_dim=action_dim)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    
    # Standard uniform replay buffer
    replay_buffer = UniformReplayBuffer(capacity=buffer_capacity)
    
    # Exploration schedule
    epsilon = epsilon_start
    # For adaptive epsilon based on performance
    reward_window = 50  # Window size for computing average reward
    target_reward = 100  # CartPole-v1 target reward
    epsilon_adjustment_rate = 0.05  # How fast epsilon adjusts to performance

    # Training tracking
    episode_rewards = []
    plot_avg_rewards = []
    
    # Train across different pole lengths (for generalization)
    pole_lengths = np.linspace(0.4, 1.8, 5)
    
    # Training loop
    for episode in range(episodes):
        # Cycle through pole lengths
        pole_len = float(pole_lengths[episode % len(pole_lengths)])
        env.unwrapped.length = pole_len
        
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(state)
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, _, __ = env.step(action)

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            
            # Store transition (no reward scaling - pure baseline)
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                train_dqn_step(q_network, target_network, replay_buffer, 
                             optimizer, batch_size, gamma)
        
        episode_rewards.append(episode_reward)
        
        # Adaptive epsilon based on recent performance
        if len(episode_rewards) >= reward_window:
            # Compute moving average reward
            avg_recent_reward = np.mean(episode_rewards[-reward_window:])

            # Normalize performance (0 to 1 scale, where 1 = target achieved)
            performance_ratio = min(avg_recent_reward / target_reward, 1.0)

            # Target epsilon based on performance (high reward -> low epsilon)
            target_epsilon = epsilon_min + (epsilon_start - epsilon_min) * (1 - performance_ratio)

            # Smoothly adjust epsilon towards target
            epsilon = epsilon + epsilon_adjustment_rate * (target_epsilon - epsilon)
        else:
            # Early episodes: gradual initial decay to build up replay buffer
            # This ensures some exploration before adaptive mechanism kicks in
            initial_decay_progress = len(episode_rewards) / reward_window
            epsilon = epsilon_start - (epsilon_start - 0.5) * initial_decay_progress

        # Clamp epsilon to valid range
        epsilon = max(epsilon_min, min(epsilon_start, epsilon))

        # Update target network periodically
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Log progress
        if episode % log_freq == 0:
            num_episodes = min(log_freq, len(episode_rewards))
            avg_reward = sum(episode_rewards[-num_episodes:]) / num_episodes
            plot_avg_rewards.append(avg_reward)
            print(f"Episode {episode:4d}, Avg Reward: {avg_reward:6.2f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return plot_avg_rewards, q_network.state_dict(), episode_rewards


if __name__ == "__main__":
    # Train with adaptive epsilon
    plot_avg_rewards, weights, episode_rewards = train_baseline_dqn(
        learning_rate=0.01,
        gamma=0.99,
        episodes=1000,
        buffer_capacity=10000,
        batch_size=32
    )
    
    # Save results using shared utilities so we dont have to rewrite code
    save_training_results(
        episode_rewards=episode_rewards,
        plot_avg_rewards=plot_avg_rewards,
        weights=weights,
        method_name="adaptive_epsilon",
        weights_dir="../weights"
    )
    
    # Compute and print statistics
    stats = compute_training_stats(episode_rewards, method_name="Adaptive Epsilon DQN")

    # Plot training curve
    plot_training_curve(
        plot_avg_rewards=plot_avg_rewards,
        method_name="Adaptive Epsilon DQN",
        save_path="../results/adaptive_epsilon_extreme/adaptive_epsilon_training_curve.png"
    )
    
    print("\nAdaptive epsilon training completed")
    print("\nResults saved in weights/ directory")
    print("\nTraining curve saved as adaptive_epsilon_training_curve.png")
