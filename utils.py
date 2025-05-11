# EE6892 Reinforcement Learning 
# Chinese Chess (Xiangqi) Utils Module

from board import init_board, board_to_state, print_board, state_to_board, is_king_captured, make_move
from moves import move_to_uci, uci_to_move, get_all_moves

# Function to play a game
def play_game(model, alpha_zero, num_moves=20):
    # Initialize the game
    board = init_board()
    state = board_to_state(board, True)  # Red goes first
    
    print("Starting a new game of Chinese Chess (Xiangqi)")
    print_board(board)
    
    move_history = []
    
    for move_num in range(num_moves):
        current_player = "Red" if state[14, 0, 0] > 0.5 else "Black"
        print(f"\nMove {move_num + 1}, {current_player} to play")
        
        # Select move
        move = alpha_zero.select_move(state, temperature=0.5)
        
        if move is None:
            print("Game over: No valid moves")
            break
        
        # Display move
        uci = move_to_uci(move)
        print(f"Selected move: {uci}")
        move_history.append(uci)
        
        # Apply move
        state = make_move(state, move)
        board = state_to_board(state)
        print_board(board)
        
        # Check for game over
        if is_king_captured(board):
            print(f"Game over: {current_player} wins (opponent's king captured)")
            break
    
    print("\nMove history:")
    for i, move in enumerate(move_history):
        print(f"{i+1}. {move}")

# Simple function to use your model interactively
def play_interactive(model, alpha_zero):
    # Initialize the game
    board = init_board()
    state = board_to_state(board, True)  # Red goes first
    
    print("Starting a new interactive game of Chinese Chess (Xiangqi)")
    print("You are playing as Black, AlphaZero is playing as Red")
    print_board(board)
    
    move_num = 0
    
    while True:
        move_num += 1
        current_player = "Red" if state[14, 0, 0] > 0.5 else "Black"
        print(f"\nMove {move_num}, {current_player} to play")
        
        if current_player == "Red":  # AI's turn
            # Select move
            print("AlphaZero is thinking...")
            move = alpha_zero.select_move(state, temperature=0.5)
            
            if move is None:
                print("Game over: No valid moves for Red")
                print("You win!")
                break
            
            # Display move
            uci = move_to_uci(move)
            print(f"AlphaZero plays: {uci}")
            
        else:  # Human's turn (Black)
            # Get valid moves
            valid_moves = get_all_moves(board, False)  # Black's turn
            valid_uci_moves = [move_to_uci(move) for move in valid_moves]
            
            # Display valid moves
            print("Valid moves:")
            for i, uci_move in enumerate(valid_uci_moves[:10]):  # Show first 10 moves
                print(f"{i+1}. {uci_move}")
            if len(valid_uci_moves) > 10:
                print(f"... and {len(valid_uci_moves) - 10} more")
            
            # Get human input
            while True:
                try:
                    move_input = input("Enter your move (e.g., 'e7-e6') or '?' for help: ")
                    
                    if move_input == '?':
                        print("Format: 'e7-e6' where e7 is the source position and e6 is the destination")
                        print("You can also enter a move number from the list above (e.g., '1' for the first listed move)")
                        continue
                    
                    # Check if input is a move number
                    try:
                        move_idx = int(move_input) - 1
                        if 0 <= move_idx < len(valid_uci_moves):
                            move_input = valid_uci_moves[move_idx]
                    except ValueError:
                        pass
                    
                    # Validate the move
                    if move_input not in valid_uci_moves:
                        print("Invalid move. Please try again.")
                        continue
                    
                    # Convert to internal move format
                    move = uci_to_move(move_input)
                    break
                except Exception as e:
                    print(f"Error: {e}. Please try again.")
        
        # Apply move
        state = make_move(state, move)
        board = state_to_board(state)
        print_board(board)
        
        # Check for game over
        if is_king_captured(board):
            winner = "Black" if current_player == "Black" else "Red"
            print(f"Game over: {winner} wins (opponent's king captured)")
            break