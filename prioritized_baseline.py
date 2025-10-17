"""
Prioritized Experience Replay DQN Implementation

- Proportional prioritization (alpha=0.6)
- Importance sampling with annealing (beta: 0.4 -> 1.0)
- ε-greedy exploration with decay
- Target network (periodic updates)
- Training across multiple pole lengths (for generalization)

Notes and a short Explanation:
This implementation uses the baseline.py structure but adds PER(prioritized experiance replay) for comparison.
Our technique (PER), samples experiances based on their priority by using TD error,instead of sampling randomly experiances from the buffer.
The main idea is: experiances with higher TD error, meaning bigger mistakes, are sampled more frequently. And by that way the agent learns faster.
Laslty, It uses importance weights to fix the bias caused by sampling some experiences more often than others.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque

from test_script import QNetwork
from utils import save_training_results, plot_training_curve, compute_training_stats


class PrioritizedReplayBuffer:
    """A simple proportional Prioritized Experience Replay buffer.

    Stores transitions in a circular buffer, samples indices proportional to priority**alpha,
    returns importance-sampling weights for bias correction, 
    and updates priorities.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        # Store cloned tensors to avoid accidental in-place modification
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        s = state.clone() if isinstance(state, torch.Tensor) else state
        ns = next_state.clone() if isinstance(next_state, torch.Tensor) else next_state
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, action, reward, ns, done))
        else:
            self.buffer[self.pos] = (s, action, reward, ns, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("The buffer is empty")

        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        sum_probs = probs.sum()
        if sum_probs == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / sum_probs

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Importance-sampling weights
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def train_dqn_step_prioritized(q_network, target_network, replay_buffer, optimizer, 
                               batch_size, gamma, beta=0.4):
    """
    Perform one training step using prioritized experience replay.
    
    Args:
        q_network: Main Q-network
        target_network: Target Q-network
        replay_buffer: PrioritizedReplayBuffer
        optimizer: Optimizer
        batch_size: Batch size
        gamma: Discount factor
        beta: Importance sampling exponent
    
    Returns:
        Loss value
    """
    # Sample a batch from prioritized replay buffer
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta=beta)

    # Current Q values (batch)
    current_q_values_all = q_network(states)
    current_q_values = current_q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Next Q values from target network
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        # Convert dones to float mask (1.0 for not-done, 0.0 for done)
        not_done = (~dones).to(dtype=torch.float32)
        target_q_values = rewards + (gamma * next_q_values * not_done)

    # Compute TD errors and weighted loss
    td_errors = target_q_values - current_q_values
    # Update priorities with absolute TD error + small eps
    new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6

    # MSE loss per sample
    loss_per_sample = td_errors.pow(2)
    weighted_loss = (weights * loss_per_sample).mean()

    optimizer.zero_grad()
    weighted_loss.backward()
    optimizer.step()

    # Update priorities in buffer
    replay_buffer.update_priorities(indices, new_priorities)
    
    return weighted_loss.item()


def train_prioritized_dqn(learning_rate=0.01, gamma=0.99, episodes=500, 
                         buffer_capacity=10000, batch_size=32,
                         epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                         target_update_freq=10, log_freq=25,
                         alpha=0.6, beta_start=0.4, beta_end=1.0):
    """
    Train a DQN with prioritized experience replay.
    
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
        alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        beta_start: Initial importance sampling weight
        beta_end: Final importance sampling weight
    
    Returns:
        (plot_avg_rewards, q_network.state_dict(), episode_rewards)
    """
    print("="*60)
    print("PRIORITIZED EXPERIENCE REPLAY DQN TRAINING")
    print("="*60)
    print("Configuration:")
    print(f"  Method: DQN with Prioritized Experience Replay")
    print(f"  Episodes: {episodes}")
    print(f"  Buffer Capacity: {buffer_capacity}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_min} (decay: {epsilon_decay})")
    print(f"  Alpha (prioritization): {alpha}")
    print(f"  Beta (IS): {beta_start} -> {beta_end}")
    print("="*60 + "\n")
    
    # Initialize environment and networks
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    q_network = QNetwork(state_dim=state_dim, action_dim=action_dim)
    target_network = QNetwork(state_dim=state_dim, action_dim=action_dim)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    
    # Prioritized replay buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=alpha)
    
    # Exploration schedule
    epsilon = epsilon_start
    
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
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            
            # Store transition (no reward scaling - pure baseline structure)
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                # Anneal beta from beta_start to beta_end
                frac = min(1.0, episode / episodes)
                beta = beta_start + frac * (beta_end - beta_start)
                train_dqn_step_prioritized(q_network, target_network, replay_buffer, 
                                         optimizer, batch_size, gamma, beta)
        
        episode_rewards.append(episode_reward)
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Log progress
        if episode % log_freq == 0:
            num_episodes = min(log_freq, len(episode_rewards))
            avg_reward = sum(episode_rewards[-num_episodes:]) / num_episodes
            plot_avg_rewards.append(avg_reward)
            frac = min(1.0, episode / episodes)
            beta = beta_start + frac * (beta_end - beta_start)
            print(f"Episode {episode:4d}, Avg Reward: {avg_reward:6.2f}, Epsilon: {epsilon:.3f}, Beta: {beta:.3f}")
    
    env.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return plot_avg_rewards, q_network.state_dict(), episode_rewards


if __name__ == "__main__":
    # Train prioritized DQN
    plot_avg_rewards, weights, episode_rewards = train_prioritized_dqn(
        learning_rate=0.01,
        gamma=0.99,
        episodes=500,
        buffer_capacity=10000,
        batch_size=32,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0
    )
    
    # Save results using shared utilities
    save_training_results(
        episode_rewards=episode_rewards,
        plot_avg_rewards=plot_avg_rewards,
        weights=weights,
        method_name="prioritized",
        weights_dir="weights"
    )
    
    # Compute and print statistics
    stats = compute_training_stats(episode_rewards, method_name="Prioritized DQN")
    
    # Plot training curve
    plot_training_curve(
        plot_avg_rewards=plot_avg_rewards,
        method_name="Prioritized DQN",
        save_path="results/prioritized_buffer/prioritized_training_curve.png"
    )
    
    print("\nPrioritized DQN training completed")
    print("\nResults saved in weights/ directory")
    print("\nTraining curve saved as prioritized_training_curve.png")
