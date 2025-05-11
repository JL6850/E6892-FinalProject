# Deep Reinforcement Learning Xiangqi Player with Monte Carlo Tree Search (DRL-Xiangqi-MCTS)

This project implements a Deep Reinforcement Learning (DRL) system for **Xiangqi (Chinese Chess)**, integrating **policy-value neural networks** with **Monte Carlo Tree Search (MCTS)** to enable self-play training, strategy learning, and real-time inference in a high-branching-factor, asymmetric, domain-specific environment.

## Overview

- **Objective**: Train an AI agent to play Xiangqi using self-play and strategic reasoning via DRL + MCTS.
- **Challenge**: Xiangqi features a 9x10 board, region-limited pieces (e.g., Advisors, Elephants), asymmetric constraints, and a larger branching factor than Western Chess.
- **Approach**:
  - Encode game states and actions into a deep neural network.
  - Use MCTS for action selection during self-play.
  - Train the model using policy gradients and value estimation based on match outcomes.

## Architecture

- **Policy-Value Network**: Outputs both move probabilities and board evaluations.
- **MCTS Integration**: Guides move selection via simulation rollouts enhanced by learned priors.
- **Self-play Loop**:
  - Play thousands of games using the current model.
  - Store move sequences, policies, and outcomes.
  - Periodically train the model from this data.
