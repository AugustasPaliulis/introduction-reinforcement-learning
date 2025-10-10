
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

def calculate_reward_scaling(pole_length, base_reward=1.0):
    optimal_length = 1.0
    difficulty = abs(pole_length - optimal_length)
    scaling_factor = 1.0 + (difficulty * 0.5)
    return base_reward * scaling_factor

def deep_q_learning(learning_rate=0.01, gamma=0.99, episodes=500):
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
    
    # Replay buffer
    replay_buffer = deque(maxlen=10000)
    batch_size = 32
    
    # Exploration parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    
    # Training tracking
    episode_rewards = []
    plot_avg_rewards = []
    
    # Training loop
    for episode in range(episodes):
        pole_length = np.random.uniform(0.4, 1.8)
        env.unwrapped.length = pole_length
        plot_avg_rewards.append(pole_length)
        
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
            scaled_reward = calculate_reward_scaling(pole_length, reward)
            # Store transition in replay buffer
            replay_buffer.append((state, action, scaled_reward, next_state, done))
            state = next_state
            episode_reward += scaled_reward
            
            # Train the network if we have enough samples
            if len(replay_buffer) >= batch_size:
                train_step(q_network, target_network, replay_buffer, optimizer, batch_size, gamma)
        
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
    return episode_rewards, q_network.state_dict(), plot_avg_rewards


def train_step(q_network, target_network, replay_buffer, optimizer, batch_size, gamma):
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.bool)

    # Current Q values from evaluation network
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q values from target network
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * ~dones)

    # Compute MSE loss
    loss = nn.MSELoss()(current_q_values, target_q_values)

    # Optimize the evaluation network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_pole_length_distribution(pole_lengths):
    plt.figure(figsize=(10, 6))
    plt.hist(pole_lengths, bins=30, alpha=0.7)
    plt.axvline(x=1.0, label='Standard pole length (1.0)')
    plt.xlabel('Pole Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Pole Lengths During Training', fontsize=14, fontweight='bold')
    plt.show()


# Hyperparameters, do not change
learning_rate = 0.01
gamma = 0.99
episodes = 500
episode_rewards, trained_weights, pole_lengths = deep_q_learning(learning_rate=learning_rate, gamma=gamma, episodes=episodes)

# Save the trained weights
torch.save(trained_weights, 'weights/strategy_adaptive_rewards.pth')
plot_pole_length_distribution(pole_lengths)
