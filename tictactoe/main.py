import sys
from typing import List, Tuple, Optional

class TicTacToe:
    """Single-player Tic-Tac-Toe game with minimax AI."""
    
    def __init__(self) -> None:
        # Board indexed 1-9, index 0 unused
        self.board: List[str] = ["", "", "", "", "", "", "", "", "", ""]
        self.human: str = "X"
        self.ai: str = "O"
    
    def print_board(self) -> None:
        """Print the board in a readable format."""
        print()
        for i in range(3):
            row = []
            for j in range(3):
                idx = i * 3 + j + 1
                cell = self.board[idx] if self.board[idx] else str(idx)
                row.append(cell)
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("-----------")
        print()
    
    def is_winner(self, player: str) -> bool:
        """Check if player has won."""
        winning_combos = [
            [1, 2, 3], [4, 5, 6], [7, 8, 9],  # rows
            [1, 4, 7], [2, 5, 8], [3, 6, 9],  # columns
            [1, 5, 9], [3, 5, 7]              # diagonals
        ]
        for combo in winning_combos:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    
    def is_board_full(self) -> bool:
        """Check if board is full."""
        return all(self.board[i] != "" for i in range(1, 10))
    
    def get_empty_cells(self) -> List[int]:
        """Return list of empty cell indices."""
        return [i for i in range(1, 10) if self.board[i] == ""]
    
    def minimax(self, is_maximizing: bool) -> int:
        """
        Minimax algorithm returning score:
        +1 if AI wins, -1 if human wins, 0 if draw.
        """
        if self.is_winner(self.ai):
            return 1
        if self.is_winner(self.human):
            return -1
        if self.is_board_full():
            return 0
        
        if is_maximizing:
            max_score = -2
            for cell in self.get_empty_cells():
                self.board[cell] = self.ai
                score = self.minimax(False)
                self.board[cell] = ""
                max_score = max(score, max_score)
            return max_score
        else:
            min_score = 2
            for cell in self.get_empty_cells():
                self.board[cell] = self.human
                score = self.minimax(True)
                self.board[cell] = ""
                min_score = min(score, min_score)
            return min_score
    
    def get_ai_move(self) -> int:
        """Find the best move for AI using minimax."""
        best_score = -2
        best_move = None
        
        for cell in self.get_empty_cells():
            self.board[cell] = self.ai
            score = self.minimax(False)
            self.board[cell] = ""
            
            if score > best_score:
                best_score = score
                best_move = cell
        
        return best_move
    
    def get_human_move(self) -> int:
        """Get valid human move (1-9, must be empty)."""
        while True:
            try:
                move = int(input("Enter your move (1-9): "))
                if move < 1 or move > 9:
                    print("Invalid: number must be between 1 and 9.")
                    continue
                if self.board[move] != "":
                    print("Invalid: cell is already occupied.")
                    continue
                return move
            except ValueError:
                print("Invalid: please enter a number between 1 and 9.")
    
    def play(self) -> None:
        """Main game loop."""
        print("Tic-Tac-Toe: You are X, AI is O")
        self.print_board()
        
        while True:
            # Human move
            human_move = self.get_human_move()
            self.board[human_move] = self.human
            print(f"You played at position {human_move}")
            self.print_board()
            
            if self.is_winner(self.human):
                print("You win! Game over.")
                return
            if self.is_board_full():
                print("Draw! Game over.")
                return
            
            # AI move
            ai_move = self.get_ai_move()
            self.board[ai_move] = self.ai
            print(f"AI played at position {ai_move}")
            self.print_board()
            
            if self.is_winner(self.ai):
                print("AI wins! Game over.")
                return
            if self.is_board_full():
                print("Draw! Game over.")
                return

if __name__ == "__main__":
    game = TicTacToe()
    game.play()
