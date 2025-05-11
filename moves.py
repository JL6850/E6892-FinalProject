#EE6892 Reinforcement Learning 
#Chinese Chess (Xiangqi) Move Setup File

#Sources : xiangqi.js and Elaine Laguerta, XiangqiGame (Python project, 2020)
# https://github.com/lengyanyu258/xiangqi.js/
# https://github.com/elaguerta/Xiangqi
# Nguyen H. Pham. “A completed implementation for Xiangqi rules.” ICGA Journal 40(3), 2018 
# https://www.researchgate.net/publication/329159969_A_completed_implementation_for_Xiangqi_rules#:~:text=A%20completed%20implementation%20for%20Xiangqi,Hong%20Pham%20%C2%B7%20Nguyen

# This file contains the scripts for our Xiangqi enviroement with utility functions
# Move encoders and movement logic for each piece

# Board Helper Functions

# is_red(piece) / is_black(piece) functions determines which side a piece belongs to
# Red pieces are shown in uppercase and black pieces are shown in lowercase
# We skip the empty boards by checking if its "." 

def is_red(piece):
        
    return piece.isupper()

def is_black(piece):
        
    return piece.islower() and piece != '.'

# is_in_board function checks if the pieces are in the board (10x9)

def is_in_board(i, j):
        
    return 0 <= i < 10 and 0 <= j < 9

#Since kings and advisors must stay in the palace we need to define the boundries
# Red palace: rows 7–9, cols 3–5
# Black palace: rows 0–2, cols 3–5

def is_in_palace(i, j, color):
        
    if color == 'red':
        return 7 <= i <= 9 and 3 <= j <= 5
    else: 
        # black
        return 0 <= i <= 2 and 3 <= j <= 5

# We check if the state is empty 

def is_empty(board, i, j):
        
    return board[i][j] == '.'

# Returns True if the piece at (i, j) is an enemy

def is_opponent(board, i, j, color):
        
    return (board[i][j] != '.' and 
            ((color == 'red' and is_black(board[i][j])) or 
             (color == 'black' and is_red(board[i][j]))))

# Move format conversions
# Converts a move (i1, j1, i2, j2) to UCI-style string like "a3-e3"

def move_to_uci(move):
        
    i1, j1, i2, j2 = move
    file_map = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i'}
        
    return f"{file_map[j1]}{10-i1}-{file_map[j2]}{10-i2}"

# Opposite of what we have done in move_to_uci function 
# Parses a string like "a3-e3" into a move tuple
# Uses 'a'..'i' to column index, and 10-i for row logic

def uci_to_move(uci):
        
    src, dst = uci.split('-')
    file_map = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8}
    src_file = file_map[src[0]]
    src_rank = int(src[1])
    dst_file = file_map[dst[0]]
    dst_rank = int(dst[1])
        
    return (9-src_rank+1, src_file, 9-dst_rank+1, dst_file)

# We need to convert a move to single integer indexes 
# So we can use them in our neural network outputs to calculate our policy loss
# First we define the moves (i1, j1) is our source poisiton whereas i2 and j2 represents the destiantion position
# The Chinese Chess board is 10x9 with gives us flattened index numbers ranging from 0 to 90

# Algorithm Concept

# Every (from, to) pair is mapped to a unique index:
# src_pos ∈ [0, 89]
# dst_pos ∈ [0, 89]

# So:

# 0 × 90 + 0 = 0 → from 0 to 0
# 0 × 90 + 1 = 1 → from 0 to 1
# 1 × 90 + 0 = 90 → from 1 to 0
# 89 × 90 + 89 = 8099 → from last to last

def move_to_action_index(move):
        
    i1, j1, i2, j2 = move
    src_pos = i1 * 9 + j1
    dst_pos = i2 * 9 + j2
        
    return src_pos * 90 + dst_pos

def action_index_to_move(index):
        
    src_pos = index // 90
    dst_pos = index % 90
    i1, j1 = src_pos // 9, src_pos % 9
    i2, j2 = dst_pos // 9, dst_pos % 9
        
    return (i1, j1, i2, j2)

# Since Xinanqi requires many advanced rules and rulebook to simplfy the architectire
# We enforced basic rules such as the King moves only one square orthogonally
# The King must stay inside its palace
# and The King can only capture opponent pieces or move to empty squares.

# However, in this implementation we have NOT included some advanced rules such as King Face-to-Face Rule
# Source : https://www.xqinenglish.com/index.php?Itemid=569&catid=119&id=923%3Athe-rules-of-xiangqi-chinese-chess&lang=en&option=com_content&view=article&
# "In Xiangqi, if the two Kings are on the same file (same column) and no pieces are between them, then neither King may move into that position" ( King Face-to-Face Rule ) 

def get_king_moves(board, i, j, color):
        
    moves = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if (is_in_board(ni, nj) and is_in_palace(ni, nj, color) and 
            (is_empty(board, ni, nj) or is_opponent(board, ni, nj, color))):
            moves.append((i, j, ni, nj))
                
    return moves

# Like we did in our get_king_moves we did the same concepts for advisor
# The Advisor is a piece that:
# Must stay inside the palace
# Can only move diagonally one step
# It exists only to protect the King and never leaves the 3×3 palace

def get_advisor_moves(board, i, j, color):
    moves = []
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if (is_in_board(ni, nj) and is_in_palace(ni, nj, color) and 
            (is_empty(board, ni, nj) or is_opponent(board, ni, nj, color))):
            moves.append((i, j, ni, nj))
                
    return moves

# The Elephant:
# Moves exactly two steps diagonally
# Cannot cross the river
# Red side: must stay on rows 
# Black side: must stay on rows 
# Can be blocked by a piece at the halfway point 

# Our main condition consists of :
# is_in_board(ni, nj) corresponds to not go off the 10×9 board
# Red: ni >= 5	makes sure that stays on Red’s side of the river
# Black: ni <= 4   makes sure that stays on Red’s side of the river
# is_empty(ei, ej)	The “elephant eye” must not be blocked so we need to check that
# is_empty or is_opponent is the destination which must be either empty or an enemy

def get_elephant_moves(board, i, j, color):
    moves = []

    # We add 4 diagnonal jumps, each consist of two blocks 
    directions = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
            
        # Elephant eye position
        ei, ej = i + di//2, j + dj//2
        if (is_in_board(ni, nj) and 
            ((color == 'red' and ni >= 5) or (color == 'black' and ni <= 4)) and 
            is_empty(board, ei, ej) and 
            (is_empty(board, ni, nj) or is_opponent(board, ni, nj, color))):
            moves.append((i, j, ni, nj))
                
    return moves

# The Horse (马 / 馬):
# Moves in an "L" shape: two steps in one direction and then one perpendicular step
# Can be blocked if the adjacent square ("horse leg") is occupied
# Does not jump over pieces like a knight in Western chess

# Our main condition consists of:
# is_in_board(ni, nj) ensures the destination is inside the 10×9 board
# is_empty or is_opponent: the destination must be either empty or hold an enemy piece
# is_empty(leg_i, leg_j) checks that the "horse leg" is not blocked


def get_horse_moves(board, i, j, color):
    moves = []

    # 8 L-shaped move options
    steps = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),  # Vertical first, then horizontal
        (1, 2), (1, -2), (-1, 2), (-1, -2)   # Horizontal first, then vertical
    ]

    for di, dj in steps:
        ni, nj = i + di, j + dj  # Final landing square

        # We have to make sure move is on the board, and target is valid
        if is_in_board(ni, nj) and (is_empty(board, ni, nj) or is_opponent(board, ni, nj, color)):

            # Then we determine the the "horse leg" square
            # This is the square directly adjacent in the first step of movement
                
            leg_i, leg_j = i, j
            if abs(di) == 2:
                    
                # Moving vertically
                # check vertical leg
                leg_i = i + di // 2  
            else:
                # Moving horizontally 
                # check horizontal leg
                leg_j = j + dj // 2  

            # Ensure the leg is not blocked
            if is_empty(board, leg_i, leg_j):
                moves.append((i, j, ni, nj))

    return moves

# The Chariot (车 / 車):
# Moves exactly like a rook in Western Chess — horizontally or vertically any number of squares
# It can be blocked by any piece and cannot jump over them
# Can capture opponent pieces by stopping on their square

# Our main conditions:
# is_in_board(ni, nj): ensures the move stays within the 10×9 board
# is_empty: chariot can keep sliding through empty squares
# is_opponent: can capture an opponent and then stop
# Otherwise (own piece blocks), we break the direction

def get_chariot_moves(board, i, j, color):
    moves = []

    # 4 orthogonal directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for di, dj in directions:
        # Try stepping further in that direction, up to the board edge
        for step in range(1, 10):
            ni = i + di * step
            nj = j + dj * step

            # If out of bounds, stop checking in this direction
            if not is_in_board(ni, nj):
                break

            if is_empty(board, ni, nj):
                # Empty square → legal move, keep going
                moves.append((i, j, ni, nj))
            elif is_opponent(board, ni, nj, color):
                # Can capture opponent piece → legal move, but stop after this
                moves.append((i, j, ni, nj))
                break
            else:
                # Own piece blocks the path → stop here
                break

    return moves

# The Cannon (炮 / 包):
# Moves like the chariot (rook) when **not capturing**
# To **capture**, it must jump exactly **one piece** (any piece) called the "platform"
# It cannot jump over more than one piece or capture without a platform in between

# Our main conditions:
# is_in_board(ni, nj): ensures within the 10×9 board
# is_empty: allows the cannon to move freely when not capturing
# Once a blocking piece is found (platform), we begin a second loop:
# We search for the next non-empty square — if it's an opponent, it's a legal capture

# The Cannon (炮 / 包):
# Moves like the chariot (rook) when **not capturing**
# To **capture**, it must jump exactly **one piece** (any piece) called the "platform"
# It cannot jump over more than one piece or capture without a platform in between

# Our main conditions:
# is_in_board(ni, nj): ensures within the 10×9 board
# is_empty: allows the cannon to move freely when not capturing
# Once a blocking piece is found (platform), we begin a second loop:
# We search for the next non-empty square — if it's an opponent, it's a legal capture

def get_cannon_moves(board, i, j, color):
    moves = []

    # Cannon moves in straight lines like rook: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for di, dj in directions:
        # Until the first block we can move freely
        for step in range(1, 10):
            ni, nj = i + di * step, j + dj * step

            # Stop if move is outside the board
            if not is_in_board(ni, nj):
                break

            if is_empty(board, ni, nj):
                # Cannon moves like a rook when not capturing
                moves.append((i, j, ni, nj))
            else:
                # Found a platform piece
                # Prepare for capture
                platform_i, platform_j = ni, nj

                # Find an opponent to capture
                for capture_step in range(1, 10):
                    ci = platform_i + di * capture_step
                    cj = platform_j + dj * capture_step

                    # Break if off board
                    if not is_in_board(ci, cj):
                        break

                    # Skip empty squares beyond the platform
                    if is_empty(board, ci, cj):
                        continue

                    # First non-empty square beyond platform
                    if is_opponent(board, ci, cj, color):
                        moves.append((i, j, ci, cj))
                            
                    # Whether it's capturable or not, we stop after the first piece
                    break

                # Stop further scanning in the current direction after platform
                break

    return moves
        
# The Pawn (兵 / 卒) in Xiangqi has some differences and it  behaves like the following:

# Before crossing the river:
#   - Can only move forward (Red → up, Black → down)

# After crossing the river (i.e. reaching enemy side):
#   - Can also move horizontally (left/right)

# Pawns never move backward
# Can only capture by moving forward or sideways — same as normal movement

# Main conditions:
# Red moves "up" the board (i decreases)
# -Black moves "down" the board (i increases)
# Once across the river (Red: i < 5, Black: i >= 5), sideways movement is unlocked
# Must stay on board and only capture opponent or move to empty square

def get_pawn_moves(board, i, j, color):
    moves = []
    if color == 'red':
        # Red pawns move up
        if i > 0:
            if is_empty(board, i-1, j) or is_opponent(board, i-1, j, color):
                moves.append((i, j, i-1, j))
        # If crossed river, can move horizontally
        if i < 5:
            for dj in [-1, 1]:
                nj = j + dj
                if is_in_board(i, nj) and (is_empty(board, i, nj) or is_opponent(board, i, nj, color)):
                    moves.append((i, j, i, nj))
    else:  # black
        # Black pawns move down
        if i < 9:
            if is_empty(board, i+1, j) or is_opponent(board, i+1, j, color):
                moves.append((i, j, i+1, j))
        # If crossed river, can move horizontally
        if i >= 5:
            for dj in [-1, 1]:
                nj = j + dj
                if is_in_board(i, nj) and (is_empty(board, i, nj) or is_opponent(board, i, nj, color)):
                    moves.append((i, j, i, nj))
    return moves

# Also, we need to get all legal moves for a specific piece at (i, j)
# This function acts as a router — it checks the piece type and color,
# then calls the corresponding move generation function


# It handles:
# Skipping empty squares ('.')
# Determining if the piece belongs to Red or Black
# Dispatching to the correct rule-based movement logic 

def get_piece_moves(board, i, j):
    piece = board[i][j]
    if piece == '.':
        return []
    
    color = 'red' if is_red(piece) else 'black'
    
    piece_lower = piece.lower()
    if piece_lower == 'k':  # King
        return get_king_moves(board, i, j, color)
    elif piece_lower == 'a':  # Advisor
        return get_advisor_moves(board, i, j, color)
    elif piece_lower == 'b':  # Elephant
        return get_elephant_moves(board, i, j, color)
    elif piece_lower == 'n':  # Horse
        return get_horse_moves(board, i, j, color)
    elif piece_lower == 'r':  # Chariot
        return get_chariot_moves(board, i, j, color)
    elif piece_lower == 'c':  # Cannon
        return get_cannon_moves(board, i, j, color)
    elif piece_lower == 'p':  # Pawn
        return get_pawn_moves(board, i, j, color)
    return []

# Get all legal moves for current player
def get_all_moves(board, red_turn):
    moves = []
    for i in range(10):
        for j in range(9):
            piece = board[i][j]
            if piece != '.':
                if (red_turn and is_red(piece)) or (not red_turn and is_black(piece)):
                    moves.extend(get_piece_moves(board, i, j))
    return moves