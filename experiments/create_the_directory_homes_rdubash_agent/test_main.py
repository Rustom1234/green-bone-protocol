import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

# Import the TicTacToe class from the embedded code
from main import TicTacToe


class TestTicTacToeInitialization:
    def test_init_creates_empty_board(self):
        game = TicTacToe()
        assert len(game.board) == 10
        assert game.board[0] == ""
        assert all(game.board[i] == "" for i in range(1, 10))
    
    def test_init_sets_players(self):
        game = TicTacToe()
        assert game.human == "X"
        assert game.ai == "O"


class TestIsWinner:
    def test_human_wins_row(self):
        game = TicTacToe()
        game.board[1] = "X"
        game.board[2] = "X"
        game.board[3] = "X"
        assert game.is_winner("X") is True
    
    def test_ai_wins_row(self):
        game = TicTacToe()
        game.board[4] = "O"
        game.board[5] = "O"
        game.board[6] = "O"
        assert game.is_winner("O") is True
    
    def test_human_wins_column(self):
        game = TicTacToe()
        game.board[1] = "X"
        game.board[4] = "X"
        game.board[7] = "X"
        assert game.is_winner("X") is True
    
    def test_ai_wins_column(self):
        game = TicTacToe()
        game.board[2] = "O"
        game.board[5] = "O"
        game.board[8] = "O"
        assert game.is_winner("O") is True
    
    def test_human_wins_diagonal(self):
        game = TicTacToe()
        game.board[1] = "X"
        game.board[5] = "X"
        game.board[9] = "X"
        assert game.is_winner("X") is True
    
    def test_ai_wins_diagonal(self):
        game = TicTacToe()
        game.board[3] = "O"
        game.board[5] = "O"
        game.board[7] = "O"
        assert game.is_winner("O") is True
    
    def test_no_winner_empty_board(self):
        game = TicTacToe()
        assert game.is_winner("X") is False
        assert game.is_winner("O") is False
    
    def test_no_winner_partial_board(self):
        game = TicTacToe()
        game.board[1] = "X"
        game.board[2] = "O"
        game.board[3] = "X"
        assert game.is_winner("X") is False
        assert game.is_winner("O") is False


class TestIsBoardFull:
    def test_empty_board_not_full(self):
        game = TicTacToe()
        assert game.is_board_full() is False
    
    def test_full_board(self):
        game = TicTacToe()
        for i in range(1, 10):
            game.board[i] = "X" if i % 2 == 0 else "O"
        assert game.is_board_full() is True
    
    def test_one_empty_cell(self):
        game = TicTacToe()
        for i in range(1, 10):
            game.board[i] = "X"
        game.board[5] = ""
        assert game.is_board_full() is False


class TestGetEmptyCells:
    def test_all_cells_empty(self):
        game = TicTacToe()
        empty = game.get_empty_cells()
        assert len(empty) == 9
        assert empty == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    def test_some_cells_occupied(self):
        game = TicTacToe()
        game.board[1] = "X"
        game.board[5] = "O"
        empty = game.get_empty_cells()
        assert 1 not in empty
        assert 5 not in empty
        assert len(empty) == 7
    
    def test_all_cells_occupied(self):
        game = TicTacToe()
        for i in range(1, 10):
            game.board[i] = "X"
        empty = game.get_empty_cells()
        assert len(empty) == 0


class TestMinimax:
    def test_minimax_ai_wins(self):
        game = TicTacToe()
        game.board[1] = "O"
        game.board[2] = "O"
        score = game.minimax(True)
        assert score == 1
    
    def test_minimax_human_wins(self):
        game = TicTacToe()
        game.board[1] = "X"
        game.board[2] = "X"
        score = game.minimax(False)
        assert score == -1
    
    def test_minimax_draw(self):
        game = TicTacToe()
        for i in range(1, 10):
            game.board[i] = "X" if i % 2 == 1 else "O"
        score = game.minimax(True)
        assert score == 0
    
    def test_minimax_maximizing_chooses_best(self):
        game = TicTacToe()
        game.board[1] = "O"
        game.board[2] = "O"
        game.board[4] = "X"
        score = game.minimax(True)
        assert score >= -1
    
    def test_minimax_minimizing_chooses_best(self):
        game = TicTacToe()
        game.board[1] = "X"
        game.board[2] = "X"
        game.board[4] = "O"
        score = game.minimax(False)
        assert score <= 1


class TestGetAIMove:
    def test_ai_blocks_human_win(self):
        game = TicTacToe()
        game.board[1] = "X"
        game.board[2] = "X"
        ai_move = game.get_ai_move()
        assert ai_move == 3
    
    def test_ai_takes_winning_move(self):
        game = TicTacToe()
        game.board[1] = "O"
        game.board[2] = "O"
        ai_move = game.get_ai_move()
        assert ai_move == 3
    
    def test_ai_move_valid(self):
        game = TicTacToe()
        ai_move = game.get_ai_move()
        assert 1 <= ai_move <= 9
        assert game.board[ai_move] == ""
    
    def test_ai_move_prefers_center(self):
        game = TicTacToe()
        ai_move = game.get_ai_move()
        assert ai_move == 5
    
    def test_ai_move_returns_valid_cell(self):
        game = TicTacToe()
        for _ in range(5):
            game.board[game.get_ai_move()] = "O"
            if len(game.get_empty_cells()) == 0:
                break
            empty_before = len(game.get_empty_cells())
            human_move = game.get_empty_cells()[