"""
Shared Utilities Module
=======================
Reusable functions for training tracking, evaluation, plotting, and results export.
All implementations (baseline, prioritized, etc.) can import from this module.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def save_training_results(episode_rewards, plot_avg_rewards, weights, 
                          method_name="baseline", weights_dir="weights"):
    """
    Save training results: weights, episode rewards, and average rewards.
    
    Args:
        episode_rewards: List of per-episode rewards
        plot_avg_rewards: List of average rewards (computed every N episodes)
        weights: Model state dict
        method_name: Name of the method (used for file naming)
        weights_dir: Directory to save weights
    
    Returns:
        Dictionary with file paths
    """
    # Create weights directory if needed
    os.makedirs(weights_dir, exist_ok=True)
    
    # Save model weights
    weights_path = os.path.join(weights_dir, f"{method_name}_weights.pth")
    torch.save(weights, weights_path)
    
    # Save episode rewards as numpy array
    rewards_path = f"{method_name}_episode_rewards.npy"
    np.save(rewards_path, np.array(episode_rewards))
    
    # Save average rewards
    avg_rewards_path = f"{method_name}_avg_rewards.npy"
    np.save(avg_rewards_path, np.array(plot_avg_rewards))
    
    print(f"✅ Saved {method_name} results:")
    print(f"   Weights: {weights_path}")
    print(f"   Episode rewards: {rewards_path}")
    print(f"   Average rewards: {avg_rewards_path}")
    
    return {
        'weights': weights_path,
        'episode_rewards': rewards_path,
        'avg_rewards': avg_rewards_path
    }


def plot_training_curve(plot_avg_rewards, method_name="baseline", save_path=None):
    """
    Plot training curve showing average reward over episodes.
    
    Args:
        plot_avg_rewards: List of average rewards
        method_name: Name of method (for title/legend)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    episodes = np.arange(0, len(plot_avg_rewards) * 25, 25)  # Assuming logging every 25 episodes
    
    plt.plot(episodes, plot_avg_rewards, linewidth=2, label=method_name)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward (over 25 episodes)', fontsize=12)
    plt.title(f'Training Progress: {method_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curve saved to {save_path}")
    
    plt.close()  # Close instead of show to avoid blocking


def compare_training_curves(results_dict, save_path=None):
    """
    Compare training curves from multiple methods.
    
    Args:
        results_dict: Dict mapping method_name -> plot_avg_rewards
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 7))
    
    for method_name, avg_rewards in results_dict.items():
        episodes = np.arange(0, len(avg_rewards) * 25, 25)
        plt.plot(episodes, avg_rewards, linewidth=2, label=method_name, marker='o', markersize=4)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward (over 25 episodes)', fontsize=12)
    plt.title('Training Progress Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to {save_path}")
    
    plt.close()  # Close instead of show to avoid blocking


def compute_training_stats(episode_rewards, method_name="method"):
    """
    Compute and print summary statistics for training run.
    
    Args:
        episode_rewards: List of per-episode rewards
        method_name: Name of method
    
    Returns:
        Dictionary with stats
    """
    rewards_array = np.array(episode_rewards)
    
    stats = {
        'method': method_name,
        'mean_reward': np.mean(rewards_array),
        'std_reward': np.std(rewards_array),
        'min_reward': np.min(rewards_array),
        'max_reward': np.max(rewards_array),
        'final_100_mean': np.mean(rewards_array[-100:]) if len(rewards_array) >= 100 else np.mean(rewards_array),
        'total_episodes': len(rewards_array)
    }
    
    print(f"\n{'='*60}")
    print(f"Training Statistics: {method_name}")
    print(f"{'='*60}")
    print(f"Total Episodes:        {stats['total_episodes']}")
    print(f"Mean Reward:           {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Min/Max Reward:        {stats['min_reward']:.1f} / {stats['max_reward']:.1f}")
    print(f"Final 100 Eps Mean:    {stats['final_100_mean']:.2f}")
    print(f"{'='*60}\n")
    
    return stats


def export_results_to_excel(stats_list, filename="training_comparison.xlsx"):
    """
    Export training statistics from multiple methods to Excel.
    
    Args:
        stats_list: List of stats dictionaries from compute_training_stats
        filename: Output Excel filename
    """
    df = pd.DataFrame(stats_list)
    df.to_excel(filename, index=False)
    print(f"📋 Results exported to {filename}")


def load_trained_weights(weights_path, state_dim, action_dim):
    """
    Load trained model weights into a QNetwork.
    
    Args:
        weights_path: Path to saved weights
        state_dim: State dimension
        action_dim: Action dimension
    
    Returns:
        Loaded QNetwork in eval mode
    """
    from test_script import QNetwork
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    
    model = QNetwork(state_dim, action_dim)
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    return model
