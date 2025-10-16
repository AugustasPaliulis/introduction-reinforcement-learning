
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time 
import matplotlib.pyplot as plt
import random
from collections import deque

from test_script import QNetwork

# This is the class I added for strategically samping from the buffer
class PrioritizedReplayBuffer:
    """A simple proportional Prioritized Experience Replay buffer.

    A small proportional prioritized replay buffer. 
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
        # store cloned tensors to avoid accidental in-place modification
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



def deep_q_learning(learning_rate, gamma, episodes, hidden_dim):
    """
    Deep Q-Learning algorithm implementation
    """
    # Initialize environment and networks
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Q-networks
    q_network = QNetwork(state_dim=state_dim, action_dim=action_dim)
    target_network = QNetwork(state_dim=state_dim, action_dim=action_dim)

    # Copy weights to target network
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    # Prioritized Replay buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
    batch_size = 32
    
    # Exploration parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    # Training tracking
    episode_rewards = []
    plot_avg_rewards = []
    
    # Training across different pole lengths: cycle through a set of lengths per episode
    pole_lengths = np.linspace(0.4, 1.8, 5)  # small set for training diversity

    # Training loop
    for episode in range(episodes):
        # pick pole length round-robin
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
            # Scale reward by pole length to emphasize harder tasks
            # baseline length ~= 0.7 (default CartPole length ~0.5-0.6); use pole_len as multiplier
            reward_scaled = reward * (pole_len / 0.7)

            # Store transition in replay buffer (store clones to avoid aliasing)
            replay_buffer.add(state, action, reward_scaled, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train the network if we have enough samples
            if len(replay_buffer.buffer) >= batch_size:
                # beta can be annealed from 0.4 -> 1.0; use simple schedule here
                frac = min(1.0, episode / episodes)
                beta = 0.4 + frac * (1.0 - 0.4)
                train_dqn(q_network, target_network, replay_buffer, optimizer, batch_size, gamma, beta)
        
        episode_rewards.append(episode_reward)
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Update target network periodically
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Track average rewards
        if episode % 25 == 0:
            num_episodes = min(25, len(episode_rewards))
            avg_reward = sum(episode_rewards[-num_episodes:]) / num_episodes
            plot_avg_rewards.append(avg_reward)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    return plot_avg_rewards, q_network.state_dict()


def train_dqn(q_network, target_network, replay_buffer, optimizer, batch_size, gamma, beta=0.4):
    """
    Train the DQN using a batch of experiences from the prioritized replay buffer.

    Uses importance-sampling weights to correct for the bias introduced by
    prioritized sampling. After learning, updates priorities using per-sample
    absolute TD errors.
    """
    # Sample a batch from prioritized replay buffer
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta=beta)

    # Current Q values (batch)
    current_q_values_all = q_network(states)
    current_q_values = current_q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Next Q values from target network
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        # convert dones to float mask (1.0 for not-done, 0.0 for done)
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


# Hyperparameters, do not change
learning_rate = 0.01
gamma = 0.99
episodes = 500
hidden_dim = 32
plot_avg_rewards, weights = deep_q_learning(learning_rate, gamma, episodes, hidden_dim)

# Save the trained weights
torch.save(weights, 'weights/dqn_weights.pth')
print("Trained weights saved to 'dqn_weights.pth'")