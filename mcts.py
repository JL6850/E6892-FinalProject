#EE6892 Reinforcement Learning 
#Chinese Chess (Xiangqi) Monte Carlo Tree Search Module

# Each MCTSNode represents a game state and stores:
# Statistics needed for search (value, visits)
# Legal move children
# Connection to its parent
# The move that led to it
import math
from board import state_to_board, make_move
from moves import get_all_moves, move_to_action_index

# Represents a node in the MCTS tree
# state is the current game state (15x10x9) tensor
# parent node is the root node
# move represents the amount of moves taken to reach this node state
# we have a dictionary to store all the moves
# We also check the visit_count to use in later stages
# Value_sum represents the sum of simulation values 
# P(s,a): prior probability from policy net
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0
    
    # Average value (Q) of this node based on rollouts
    def value(self):
        if self.visit_count == 0:
            # If unvisited, return neutral value
            return 0
        return self.value_sum / self.visit_count
    
    # In AlphaZero-style algorithms, the UCB score is used during the selection phase to decide which child node to explore next
    # By using the UCB formula we balance the explotation and exploration 
    # The idea is not to choose the move that looks the best and try other moves to recevice high potential
    # UCB = Q + U = (value_sum / visit_count) + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
    def select_child(self, c_puct=1.0):
        # Select child with highest UCB value
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            # UCB formula
            ucb = child.value() + c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
                
            if ucb > best_score:
                best_score = ucb
                best_child = child
                
        return best_child
    
    def expand(self, priors, moves):
        # Expands this node by adding child nodes for all legal moves,
        # using prior probabilities (from the policy network) to guide search.
        for move, prior in zip(moves, priors):
            # For each legal move and its corresponding prior probability:
            
            if move not in self.children:
                # Only add a child node if this move hasn't been explored yet
                
                # Apply the move to the current game state to get the new state
                child_state = make_move(self.state, move)
                # Create a new child node for that resulting state
                self.children[move] = MCTSNode(
                    child_state,       # the new game state
                    parent=self,       # link back to the current node (for backpropagation)
                    move=move          # the move that led to this child
                )
                    
                # Assign the policy prior (from neural net) to guide future selection
                self.children[move].prior = prior
    
    def update(self, value):
        # This method is called during backpropagation.
        # After a simulation (playout) ends and returns a value,
        # we use it to update this node's statistics.
            
        self.visit_count += 1       # Increase the number of times this node was visited (N(s))
        self.value_sum += value     # Accumulate the simulation result (to later compute average Q(s))