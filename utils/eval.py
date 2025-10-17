import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from test_script import QNetwork
import os
import scipy.stats as stats


def load_trained_model(weights_path='dqn_weights.pth'):
    """Load trained model from checkpoint"""
    # Check both current directory and weights subdirectory
    if not os.path.exists(weights_path):
        weights_in_dir = os.path.join('../weights', weights_path)
        if os.path.exists(weights_in_dir):
            weights_path = weights_in_dir
        else:
            raise FileNotFoundError(f"Could not find weights at {weights_path}")

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    model = QNetwork(state_dim, action_dim)
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


def test_on_pole_length(model, pole_length, num_episodes=10, max_steps=500):
    """Test model on specific pole length"""
    env = gym.make('CartPole-v1')
    env.unwrapped.length = pole_length

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0

        for step in range(max_steps):
            with torch.no_grad():
                q_values = model(state)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                break

            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward

        episode_rewards.append(total_reward)

    env.close()
    return episode_rewards


def evaluate_strategy(weights_path='dqn_weights.pth', baseline_weights_path='baseline_weights.pth', method_name='Trained Model'):
    """Evaluate model across different pole lengths against a baseline model with statistical testing"""
    print(f"Evaluating {method_name} against Baseline Model")
    print("=" * 60)

    # Load models
    try:
        trained_model = load_trained_model(weights_path)
        print(f"Loaded model from {weights_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run training script first to generate weights")
        return

    try:
        baseline_model = load_trained_model(baseline_weights_path)
        print(f"Loaded baseline model from {baseline_weights_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Baseline weights not found. Please provide valid weights for comparison.")
        return

    # Test across pole lengths
    pole_lengths = np.linspace(0.4, 1.8, 15)
    num_test_episodes = 10

    print(f"Testing {len(pole_lengths)} different pole lengths ({num_test_episodes} episodes each)")

    results = {
        'pole_length': [],
        'trained_mean': [],
        'trained_std': [],
        'baseline_mean': [],
        'baseline_std': [],
        'improvement_ratio': [],
        'trained_scores': [],  # Store all trained scores
        'baseline_scores': []   # Store all baseline scores
    }

    for i, pole_len in enumerate(pole_lengths):
        print(f"Testing pole length {pole_len:.2f} ({i+1}/{len(pole_lengths)})")

        trained_scores = test_on_pole_length(trained_model, pole_len, num_test_episodes)
        trained_mean = np.mean(trained_scores)
        trained_std = np.std(trained_scores)

        baseline_scores = test_on_pole_length(baseline_model, pole_len, num_test_episodes)
        baseline_mean = np.mean(baseline_scores)
        baseline_std = np.std(baseline_scores)

        improvement = trained_mean / max(baseline_mean, 1.0)

        results['pole_length'].append(pole_len)
        results['trained_mean'].append(trained_mean)
        results['trained_std'].append(trained_std)
        results['baseline_mean'].append(baseline_mean)
        results['baseline_std'].append(baseline_std)
        results['improvement_ratio'].append(improvement)
        results['trained_scores'].extend(trained_scores)  # Append scores
        results['baseline_scores'].extend(baseline_scores)   # Append scores

    create_evaluation_plots(results, method_name)
    print_evaluation_summary(results, method_name)

    return results


def create_evaluation_plots(results, method_name='Trained Model'):
    """Generate evaluation plots"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Adjusted for 2 subplots

    pole_lengths = results['pole_length']

    # Performance comparison
    axes[0].errorbar(pole_lengths, results['trained_mean'], yerr=results['trained_std'],
                        label=method_name, marker='o', capsize=4, linewidth=2)
    axes[0].errorbar(pole_lengths, results['baseline_mean'], yerr=results['baseline_std'],
                        label='Baseline', marker='s', capsize=4, linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Pole Length')
    axes[0].set_ylabel('Episode Length')
    axes[0].set_title('Performance Across Pole Lengths')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Improvement ratio
    axes[1].plot(pole_lengths, results['improvement_ratio'], 'g-o', linewidth=2)
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Pole Length')
    axes[1].set_ylabel('Improvement Ratio')
    axes[1].set_title('Trained vs Baseline Performance')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    filename = f"{method_name.lower().replace(' ', '_')}_evaluation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plots saved as '{filename}'")
    plt.close()


def print_evaluation_summary(results, method_name='Trained Model'):
    """Print evaluation summary"""
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY: {method_name}")
    print("="*60)

    # Overall performance
    overall_trained = np.mean(results['trained_mean'])
    overall_baseline = np.mean(results['baseline_mean'])
    overall_improvement = overall_trained / overall_baseline

    print(f"\nOverall Performance:")
    print(f"  {method_name}: {overall_trained:.1f} +/- {np.mean(results['trained_std']):.1f}")
    print(f"  Baseline Model: {overall_baseline:.1f} +/- {np.mean(results['baseline_std']):.1f}")
    print(f"  Improvement: {overall_improvement:.1f}x better than baseline")

    # Statistical significance (overall)
    trained_scores = results['trained_scores']
    baseline_scores = results['baseline_scores']
    t_statistic, p_value = stats.ttest_ind(trained_scores, baseline_scores)

    print(f"\nOverall Statistical Significance:")
    print(f"  P-value: {p_value:.10f}")
    if p_value < 0.05:
        print("  The difference is statistically significant (p < 0.05)")
    else:
        print("  The difference is not statistically significant (p >= 0.05)")

    # Best and worst
    best_idx = np.argmax(results['improvement_ratio'])
    worst_idx = np.argmin(results['improvement_ratio'])

    print(f"\nBest Performance:")
    print(f"  Pole length {results['pole_length'][best_idx]:.2f}")
    print(f"  {results['improvement_ratio'][best_idx]:.1f}x better than baseline")

    print(f"\nMost Challenging:")
    print(f"  Pole length {results['pole_length'][worst_idx]:.2f}")
    print(f"  {results['improvement_ratio'][worst_idx]:.1f}x better than baseline")

    # Performance analysis
    print(f"\nStrategy Insights:")

    short_poles = [i for i, length in enumerate(results['pole_length']) if length < 1.0]
    long_poles = [i for i, length in enumerate(results['pole_length']) if length > 1.4]

    if short_poles and long_poles:
        short_improvement = np.mean([results['improvement_ratio'][i] for i in short_poles])
        long_improvement = np.mean([results['improvement_ratio'][i] for i in long_poles])

        if long_improvement > short_improvement:
            print(f"  - Better on longer poles ({long_improvement:.1f}x vs {short_improvement:.1f}x)")
            print(f"    Suggests the model handles harder tasks well")
        else:
            print(f"  - Better on shorter poles ({short_improvement:.1f}x vs {long_improvement:.1f}x)")
            print(f"    Performance decreases with task difficulty")

    improvement_std = np.std(results['improvement_ratio'])
    if improvement_std < 0.5:
        print(f"  - Consistent across pole lengths (std: {improvement_std:.2f})")
    else:
        print(f"  - Variable across pole lengths (std: {improvement_std:.2f})")

    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare reward scaling strategy against a baseline')
    parser.add_argument('--weights', type=str, default='reward_scaling_weights.pth',
                       help='Path to model weights')
    parser.add_argument('--baseline_weights', type=str, default='baseline_weights.pth',
                       help='Path to baseline model weights')
    parser.add_argument('--method', type=str, default='Reward scaling strategy',
                       help='Method name for plots')

    args = parser.parse_args()

    evaluate_strategy(weights_path=args.weights, baseline_weights_path=args.baseline_weights, method_name=args.method)
