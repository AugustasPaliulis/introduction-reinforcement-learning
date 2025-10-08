
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
            
            # Store transition in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            
            state = next_state
            episode_reward += reward
            
            # Train the network if we have enough samples
            if len(replay_buffer) >= batch_size:
                train_dqn(q_network, target_network, replay_buffer, optimizer, batch_size, gamma)
        
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


def train_dqn(q_network, target_network, replay_buffer, optimizer, batch_size, gamma):
    """
    Train the DQN using a batch of experiences from the replay buffer
    """
    # Sample a batch from replay buffer
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert to tensors
    states = torch.cat(states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.bool)
    
    # Current Q values
    current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Next Q values from target network
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + (gamma * next_q_values * ~dones)
    
    # Compute loss
    loss = nn.MSELoss()(current_q_values, target_q_values)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Hyperparameters, do not change
learning_rate = 0.01
gamma = 0.99
episodes = 500
hidden_dim = 32
plot_avg_rewards, weights = deep_q_learning(learning_rate, gamma, episodes, hidden_dim)

# Save the trained weights
torch.save(weights, 'dqn_weights.pth')
print("Trained weights saved to 'dqn_weights.pth'")