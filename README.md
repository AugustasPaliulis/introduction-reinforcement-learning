# Deep Q-Network Learning Strategies for CartPole

**Introduction to Reinforcement Learning - Assignment 3 - Group 9**

This repository implements and compares four DQN learning strategies for the OpenAI Gymnasium CartPole-v1 environment with training across multiple pole lengths.

## Table of Contents

- [Overview](#overview)
- [Learning Strategies](#learning-strategies)
- [Environment Setup](#environment-setup)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)

## Overview

The CartPole environment is a classic reinforcement learning problem where an agent must balance a pole on a moving cart. This project implements four DQN training strategies:

1. **Baseline DQN** - Standard DQN with uniform experience replay and random pole lengths
2. **Prioritized Experience Replay** - Samples important experiences more frequently
3. **Adaptive Epsilon Strategy** - Dynamically adjusts exploration based on performance
4. **Reward Scaling Strategy** - Scales rewards based on task difficulty

All implementations train across multiple pole lengths (0.4m to 1.8m) to ensure generalization beyond the default 1.0m pole length.

## Learning Strategies

### 1. Baseline DQN

Standard DQN implementation with uniform experience replay buffer and epsilon-greedy exploration with exponential decay. Serves as the comparison benchmark.

### 2. Prioritized Experience Replay

Uses prioritized sampling based on TD error magnitude. Experiences with larger prediction errors are sampled more frequently, with importance sampling weights to correct bias.

- Priority: `|TD_error| + ε`
- Alpha (α) = 0.6: Controls prioritization strength
- Beta (β) = 0.4 → 1.0: Annealing schedule for importance sampling

### 3. Adaptive Epsilon Strategy

Dynamically adjusts exploration rate based on recent performance over a 50-episode window. High performance decreases epsilon (more exploitation), while low performance increases epsilon (more exploration).

### 4. Reward Scaling Strategy

Scales rewards based on pole length difficulty. Easy configurations near 1.0m receive normal rewards, while difficult configurations (0.4m or 1.8m) receive boosted rewards to encourage generalization.

Formula: `scaled_reward = base_reward * (1.0 + |pole_length - 1.0| * 0.5)`

## Environment Setup

### Prerequisites

- Python 3.10
- Conda package manager

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/AugustasPaliulis/introduction-reinforcement-learning.git
cd introduction-reinforcement-learning
```

2. **Create the Conda environment:**

```bash
conda env create -f environment.yml
```

3. **Activate the environment:**

```bash
conda activate reinforcement_learning
```

### Key Dependencies

- PyTorch: Deep learning framework
- Gymnasium: RL environments
- NumPy: Numerical computations
- Matplotlib: Visualization
- Pandas: Data analysis

## How to Use

### Training a Model

Run the desired training script:

```bash
python training-strategies/baseline.py
python training-strategies/prioritized_baseline.py
python training-strategies/adaptive_epsilon.py
python training-strategies/strategy_reward_scaling.py
```

Training output includes model weights saved to `weights/`, episode rewards as NumPy arrays, and training statistics.

### Evaluating a Model

```python
from utils.eval import evaluate_strategy

evaluate_strategy(
    weights_path='weights/prioritized_weights.pth',
    baseline_weights_path='weights/baseline_weights.pth',
    method_name='Prioritized Experience Replay'
)
```

## Project Structure

```
introduction-reinforcement-learning/
├── training-strategies/
│   ├── baseline.py
│   ├── prioritized_baseline.py
│   ├── adaptive_epsilon.py
│   └── strategy_reward_scaling.py
├── utils/
│   ├── utils.py
│   └── eval.py
├── weights/
├── results/
├── test_script.py
└── environment.yml
```

## Authors

Group 9 - Introduction to Reinforcement Learning Course, BSc Artificial Intelligence,  VU Amsterdam
