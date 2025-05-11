# EE6892 Reinforcement Learning 
# Chinese Chess (Xiangqi) AlphaZero Network Module

# Sources : NeymarL/ChineseChess-AlphaZero  (https://github.com/NeymarL/ChineseChess-AlphaZero)
# Surag Nair. "A Simple Alpha(Go) Zero Tutorial  (https://suragnair.github.io/posts/alphazero.html)
# OpenSpiel – AlphaZero Implementation (https://openspiel.readthedocs.io/en/stable/alpha_zero.html)

# Designed to work with a neural network that outputs (policy, value)
# The key workflow: Predict → Expand root → Simulate → Backpropagate → Return move probabilities

import numpy as np
import torch

from board import state_to_board, is_king_captured, make_move
from moves import get_all_moves, move_to_action_index, action_index_to_move
from mcts import MCTSNode

# Simple AlphaZero MCTS implementation
class AlphaZero:
    def __init__(self, model, num_simulations=100):
        """
        Initialize the AlphaZero agent.

        Args:
            model: A PyTorch model that outputs (policy_logits, value) given a state
            num_simulations: Number of MCTS simulations to perform for each move
        """
        # The NN used for policy and value prediction
        self.model = model       
        # How many simulations to run per move
        self.num_simulations = num_simulations    

# The predict() function is the entry point for the game environment to the neural network
# Accepting as input a current game state represented as a NumPy tensor with shape (15, 10, 9),
# it prepares the input to the model by converting it to a PyTorch tensor and adding a batch dimension (because most neural networks expect input with shape [batch_size, channels, height, width])

    def predict(self, state):
        """
        Predict policy and value from the current state using the neural network.

        Args:
            state: The current game state (15x10x9 tensor as a NumPy array).

        Returns:
            policy: A probability distribution over all 8100 possible actions.
            value_scalar: A scalar value estimate for the current state.
        """
        # Get policy and value from the neural network
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        
        policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
        value_scalar = value.item()
        
        return policy, value_scalar
            
# get_move_probabilities is one of the core functions of our AlphaZero implementation
# It performs a complete iteration of Monte Carlo Tree Search (MCTS) from a given game position
# It begins by creating a root MCTSNode for the current game position
# Secondly, it uses the helper functions get_all_moves() and state_to_board(), 
# to return the list of all legal moves for the present player 
# If no legal moves are available, the game is finished and the function returns None

# Then our neural net comes into play, the neural network through predict() to obtains the policy logits and value
# It retrieves the prior probabilities for the legal moves from the entire action space (8100 actions), utilizing move_to_action_index() to determine each move's index
# These priors are normalized and used to expand the root node, assigning probabilities to its children

# Then the process goes into the simulation loop
# Running it num_simulations times. Within each simulation, it runs a selection step by traversing the tree from the root to a leaf according to the UCB formula

# When it hits a leaf node (a node without children), it checks whether the game is over using is_king_captured()
# Otherwise, it recursively expands the node by forming all potential child moves, retrieves their value and prior from the neural network, and adds them to the node

# If the game is over (a king was captured), it assigns a terminal value of -1.0 to the node
# If it is a draw (no legal moves), it assigns 0.0

# Once a value is computed, it backpropagates this result along the saved path

# Finally, after all simulations have been performed, the method builds a full-size action probability array (action_probs) of size 8100
# It fills visit counts for every child node of the root, and then normalizes these counts to form a probability distribution
# These probabilities are used by select_move() to decide which move to make, either deterministically or by sampling depending on the temperature parameter

    def get_move_probabilities(self, state):
        """
        Run MCTS simulations starting from the given state to generate action probabilities.

        Steps:
        1. Create root node from current state.
        2. Expand root using legal moves and policy network priors.
        3. Run multiple simulations:
            - Traverse the tree via UCB until a leaf is reached
            - Expand the leaf using NN output
            - Backpropagate the value up the search path
        4. Return action probabilities proportional to visit counts.
        """
        # Build a search tree and return move probabilities
        root = MCTSNode(state)
        
        # Get valid moves
        board = state_to_board(state)
        red_to_move = state[14, 0, 0] > 0.5
        valid_moves = get_all_moves(board, red_to_move)
        
        if not valid_moves:
            return None  # Game over
        
        # Get neural network's policy output
        policy, _ = self.predict(state)
        
        # Extract probabilities for valid moves
        valid_action_indices = [move_to_action_index(move) for move in valid_moves]
        valid_priors = policy[valid_action_indices]
        valid_priors = valid_priors / np.sum(valid_priors)  # Normalize
        
        # Expand the root with all valid moves
        root.expand(valid_priors, valid_moves)
        
        # Perform MCTS simulations
        for _ in range(self.num_simulations):
            # Selection
            node = root
            search_path = [node]
            
            # Traverse tree until we reach a leaf
            while node.children:
                node = node.select_child()
                search_path.append(node)
            
            # Check if the game is over
            board = state_to_board(node.state)
            red_to_move = node.state[14, 0, 0] > 0.5
            game_over = is_king_captured(board)
            
            if not game_over:
                # Expansion: Get neural network prediction
                valid_moves = get_all_moves(board, red_to_move)
                
                if valid_moves:
                    # Get neural network's policy output
                    policy, value = self.predict(node.state)
                    
                    # Extract probabilities for valid moves
                    valid_action_indices = [move_to_action_index(move) for move in valid_moves]
                    valid_priors = policy[valid_action_indices]
                    valid_priors = valid_priors / np.sum(valid_priors)  # Normalize
                    
                    # Expand the node
                    node.expand(valid_priors, valid_moves)
                else:
                    # No valid moves (stalemate)
                    value = 0.0
            else:
                # Game over (king captured)
                value = -1.0  # Loss for current player
            
            # Backpropagate
            for node in reversed(search_path):
                node.update(-value)  # Negative because value is from perspective of other player
                value = -value  # Flip value for next node
        
        # Return the action probabilities based on visit counts
        action_probs = np.zeros(8100)  # All possible actions
        for move, child in root.children.items():
            action_idx = move_to_action_index(move)
            action_probs[action_idx] = child.visit_count
        
        # Normalize
        action_probs = action_probs / np.sum(action_probs)
        
        return action_probs
    
    def select_move(self, state, temperature=0.0):
        """
        Choose a move from the action probabilities output by MCTS.

        Args:
            state: The current game state.
            temperature: Controls exploration (0 = greedy, >0 = stochastic sampling).

        Returns:
            move: The selected move as a (i1, j1, i2, j2) tuple.
        """
        # Get move probabilities from MCTS
        action_probs = self.get_move_probabilities(state)
        
        if action_probs is None:
            return None  # Game over
        
        # Select move based on temperature
        if temperature == 0:
            # Deterministic: choose the move with highest probability
            action_idx = np.argmax(action_probs)
        else:
            # Apply temperature and sample
            action_probs = action_probs ** (1.0 / temperature)
            action_probs = action_probs / np.sum(action_probs)
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        # Convert action index to move
        move = action_index_to_move(action_idx)
        return move