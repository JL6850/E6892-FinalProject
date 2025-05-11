#EE6892 Reinforcement Learning 
#Chinese Chess (Xiangqi) Board Setup File

#This file is for our Chinese Chess (Xiangqi) enviroement for:
# Piece encodings
# Initial board setup
# Board visualization
# Conversion between board ↔ tensor state
# Game logic /  move application and king-checking


#Importing the necessary modules
import numpy as np



# We first need to define every piece with its corresponding numbers
# Since we have 15 channels in our network which is explained in aznet.py file
# We can map each piece with a seperate channel 

# Alpha-Zero model ( Source : https://arxiv.org/abs/1712.01815 ) represents a board state as a multi-channel tensor
# [num_channels, board_height, board_width] in our case it is [15, 10, 9] 
# Piece_to_channel maps each piecce with a number so we can represent 1 if a piece's location

piece_to_channel = {

    # King (将) : 0  R - 7 B
    # Advisor (士) : 1 R - 8 B 
    # Elephant (象) : 2 R - 9 B
    # Knight (马) : 3 R - 10 B
    # Rook (车) : 4 R - 11 B
    # Cannon (炮) : 5 R - 12 B
    # Pawn (卒): 6 R - 13 B
        
    'K': 0, 'A': 1, 'B': 2, 'N': 3, 'R': 4, 'C': 5, 'P': 6,   # Red pieces
    'k': 7, 'a': 8, 'b': 9, 'n':10, 'r':11, 'c':12, 'p':13    # Black pieces
}

# Initialize the board
# We first create an empty board which we do it by initalizing a 10x9 matrix with dot to represent empty squares
# Thats because Xiangqi is played in 10x9 environment
# Our starting formation is Rook - Knight - Elephant - Advisor - King forming R N B A K A B N R in symmetric r n b a k a b n r 
# Then we have cannons located at columns 1 and 5 . C . . . C . . / . c . . . c . . . 
# And then we put the pawns 

#Board reprensentation
  # 0: r n b a k a b n r
  # 1: . . . . . . . . .
  # 2: . c . . . c . . .
  # 3: p . p . p . p . p
  # 4: . . . . . . . . .
  # 5: . . . . . . . . .
  # 6: P . P . P . P . P
  # 7: . C . . . C . . .
  # 8: . . . . . . . . .
  # 9: R N B A K A B N R

def init_board():
    board = [["." for _ in range(9)] for _ in range(10)]
    board[0] = list("rnbakabnr")
    board[2] = list(".c...c...")
    board[3] = list("p.p.p.p.p")
    board[6] = list("P.P.P.P.P")
    board[7] = list(".C...C...")
    board[9] = list("RNBAKABNR")
    return board


# for debugging and visualizaiton purposes
# Source : https://stackoverflow.com/questions/10903176/how-to-print-a-board-in-python?
def print_board(board):
    """Print the Chinese Chess board"""
        
    piece_symbols = {
        '.': ' . ',
        'K': ' K ', 'A': ' A ', 'B': ' B ', 'N': ' N ', 'R': ' R ', 'C': ' C ', 'P': ' P ',
        'k': ' k ', 'a': ' a ', 'b': ' b ', 'n': ' n ', 'r': ' r ', 'c': ' c ', 'p': ' p '
    }
    
    print("    a   b   c   d   e   f   g   h   i")
    print("  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐")
    
    for i in range(10):
        row_num = 10 - i
        print(f"{row_num} │", end="")
        
        for j in range(9):
            print(piece_symbols[board[i][j]], end="│")
        
        if i == 4:
            print("\n  ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")
        elif i < 9:
            print("\n  ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")
        else:
            print("\n  └───┴───┴───┴───┴───┴───┴───┴───┴───┘")
                
# We need to convert the board into 15-channel tensor state as we discussed previosuly
# So first we initalize a three dimensional tensor with a shape (15, 10, 9) where 15 is the total channels and 10,9 is our Chinese Chess board size
# We loop over the board and get the characters for each place
# If there is a piece in that place we use our piece to channel function to find the correct tensor shape
# Then we set the channel at state (i,j) to 1 which where start one-hot encoding teh piece locations

# We do this for 13 channels ( + 1 self channel ) but our 15th channel has a different purpose
# 15th channel is a 10x9 plane full of 1's if its RED turn and 0 otherwise
# This implementation helps our algorithm to know whose move it is 

# Multi-channel binary encoding preserves all key information:
# What pieces are where
# Who owns which piece
# Whose turn it is

def board_to_state(board, red_to_move=True):
    state = np.zeros((15, 10, 9), dtype=np.float32)

    # Define the board shape
    for i in range(10):
        for j in range(9):
            #For the specific piece 
            piece = board[i][j]
            if piece != '.':
                chan = piece_to_channel[piece]
                state[chan, i, j] = 1
    state[14, :, :] = 1 if red_to_move else 0
    return state

# state_to_board function is the exact reverse of our board_state_funciton
# The function takes a tensor which has a dimension of 15x10x9 and build a board with that information

def state_to_board(state):
    board = [["." for _ in range(9)] for _ in range(10)]
    pieces = "KABNRCPkabnrcp"
    for chan, piece in enumerate(pieces):
        if chan >= 14:
            break
        for i in range(10):
            for j in range(9):
                if state[chan, i, j] > 0.5:
                    board[i][j] = piece
    return board

# Defining the Rules
# We need to define a game-over condition checker in Xiangqi
# The function is_king_captured returns 1 if the King has been captured, 0 otherwise
# The game ends immediately if a King ('K' for Red, 'k' for Black) is captured

def is_king_captured(board):
    red_king_found = False
    black_king_found = False
    
    for i in range(10):
        for j in range(9):
            if board[i][j] == 'K':
                red_king_found = True
            elif board[i][j] == 'k':
                black_king_found = True
    
    return not (red_king_found and black_king_found)

# We need to apply a move, get the next state and flip the board
# That's why we define a make_move function which helps us 

def make_move(state, move):
        
    board = state_to_board(state)
    red_to_move = state[14, 0, 0] > 0.5
    
    # Apply move
    i1, j1, i2, j2 = move
    board[i2][j2] = board[i1][j1]
    board[i1][j1] = '.'
    
    # Return new state with updated player
    return board_to_state(board, not red_to_move)