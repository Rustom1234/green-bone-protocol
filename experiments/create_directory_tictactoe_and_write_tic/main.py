import os
import sys
from typing import Optional

BOARD_SIZE = 9
WINNING_LINES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
]


def print_board(board: list[str]) -> None:
    """Print the 3x3 board with separators."""
    print()
    for i in range(3):
        row = []
        for j in range(3):
            index = i * 3 + j
            cell = board[index] if board[index] != "" else str(index + 1)
            row.append(cell)
        print(f" {row[0]} | {row[1]} | {row[2]} ")
        if i < 2:
            print("-----------")
    print()


def check_winner(board: list[str]) -> Optional[str]:
    """Check if there's a winner. Returns 'X', 'O', or None."""
    for line in WINNING_LINES:
        if board[line[0]] == board[line[1]] == board[line[2]] != "":
            return board[line[0]]
    return None


def is_board_full(board: list[str]) -> bool:
    """Check if the board is full."""
    return all(cell != "" for cell in board)


def get_available_moves(board: list[str]) -> list[int]:
    """Get list of available move positions."""
    return [i for i in range(BOARD_SIZE) if board[i] == ""]


def minimax(
    board: list[str], depth: int, is_maximizing: bool
) -> int:
    """
    Minimax algorithm.
    Returns: +1 if AI (O) wins, -1 if Human (X) wins, 0 for draw.
    """
    winner = check_winner(board)
    
    if winner == "O":
        return 1
    elif winner == "X":
        return -1
    elif is_board_full(board):
        return 0
    
    if is_maximizing:
        max_eval = -float('inf')
        for move in get_available_moves(board):
            board[move] = "O"
            eval_score = minimax(board, depth + 1, False)
            board[move] = ""
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_available_moves(board):
            board[move] = "X"
            eval_score = minimax(board, depth + 1, True)
            board[move] = ""
            min_eval = min(min_eval, eval_score)
        return min_eval


def find_best_move(board: list[str]) -> int:
    """Find the best move for AI using minimax."""
    best_score = -float('inf')
    best_move = -1
    
    for move in get_available_moves(board):
        board[move] = "O"
        score = minimax(board, 0, False)
        board[move] = ""
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move


def main() -> None:
    """Main game loop."""
    board: list[str] = ["" for _ in range(BOARD_SIZE)]
    
    print("Welcome to Tic-Tac-Toe!")
    print("You are X, AI is O. You go first.")
    print("Enter positions 1-9 to make your move.")
    print()
    
    print_board(board)
    
    while True:
        # Human's turn (X)
        while True:
            try:
                move_input = input("Enter your move (1-9): ").strip()
                move = int(move_input) - 1
                
                if move < 0 or move >= BOARD_SIZE:
                    print("Invalid input. Please enter a number between 1 and 9.")
                    continue
                
                if board[move] != "":
                    print("That position is already taken. Try another.")
                    continue
                
                board[move] = "X"
                break
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 9.")
        
        print_board(board)
        
        # Check if human won
        winner = check_winner(board)
        if winner == "X":
            print("You win! Congratulations!")
            break
        
        # Check if it's a draw
        if is_board_full(board):
            print("It's a draw!")
            break
        
        # AI's turn (O)
        print("AI is thinking...")
        ai_move = find_best_move(board)
        board[ai_move] = "O"
        print(f"AI plays position {ai_move + 1}")
        
        print_board(board)
        
        # Check if AI won
        winner = check_winner(board)
        if winner == "O":
            print("AI wins! Better luck next time.")
            break
        
        # Check if it's a draw
        if is_board_full(board):
            print("It's a draw!")
            break


if __name__ == "__main__":
    # For automated testing, we'll simulate some moves
    import io
    
    # Test with automated input
    test_input = "5\n1\n9\n3\n7\n"
    
    # Create test by running with simulated input
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(test_input)
    try:
        main()
    finally:
        sys.stdin = old_stdin