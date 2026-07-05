import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
from main import (
    BOARD_SIZE,
    WINNING_LINES,
    print_board,
    check_winner,
    is_board_full,
    get_available_moves,
    minimax,
    find_best_move,
)


class TestBoardConstants:
    """Test board constants."""
    
    def test_board_size(self):
        assert BOARD_SIZE == 9
    
    def test_winning_lines_count(self):
        assert len(WINNING_LINES) == 8
    
    def test_winning_lines_rows(self):
        assert [0, 1, 2] in WINNING_LINES
        assert [3, 4, 5] in WINNING_LINES
        assert [6, 7, 8] in WINNING_LINES
    
    def test_winning_lines_columns(self):
        assert [0, 3, 6] in WINNING_LINES
        assert [1, 4, 7] in WINNING_LINES
        assert [2, 5, 8] in WINNING_LINES
    
    def test_winning_lines_diagonals(self):
        assert [0, 4, 8] in WINNING_LINES
        assert [2, 4, 6] in WINNING_LINES


class TestPrintBoard:
    """Test print_board function."""
    
    def test_print_empty_board(self, capsys):
        board = ["" for _ in range(9)]
        print_board(board)
        captured = capsys.readouterr()
        assert "1" in captured.out
        assert "9" in captured.out
        assert "-----------" in captured.out
    
    def test_print_board_with_moves(self, capsys):
        board = ["X", "", "O", "", "", "", "", "", ""]
        print_board(board)
        captured = capsys.readouterr()
        assert "X" in captured.out
        assert "O" in captured.out
        assert "2" in captured.out
    
    def test_print_full_board(self, capsys):
        board = ["X", "O", "X", "O", "X", "O", "O", "X", "X"]
        print_board(board)
        captured = capsys.readouterr()
        assert captured.out.count("X") == 5
        assert captured.out.count("O") == 4


class TestCheckWinner:
    """Test check_winner function."""
    
    def test_no_winner_empty_board(self):
        board = ["" for _ in range(9)]
        assert check_winner(board) is None
    
    def test_x_wins_first_row(self):
        board = ["X", "X", "X", "", "", "", "", "", ""]
        assert check_winner(board) == "X"
    
    def test_o_wins_second_row(self):
        board = ["", "", "", "O", "O", "O", "", "", ""]
        assert check_winner(board) == "O"
    
    def test_x_wins_third_row(self):
        board = ["", "", "", "", "", "", "X", "X", "X"]
        assert check_winner(board) == "X"
    
    def test_x_wins_first_column(self):
        board = ["X", "", "", "X", "", "", "X", "", ""]
        assert check_winner(board) == "X"
    
    def test_o_wins_second_column(self):
        board = ["", "O", "", "", "O", "", "", "O", ""]
        assert check_winner(board) == "O"
    
    def test_x_wins_third_column(self):
        board = ["", "", "X", "", "", "X", "", "", "X"]
        assert check_winner(board) == "X"
    
    def test_x_wins_main_diagonal(self):
        board = ["X", "", "", "", "X", "", "", "", "X"]
        assert check_winner(board) == "X"
    
    def test_o_wins_anti_diagonal(self):
        board = ["", "", "O", "", "O", "", "O", "", ""]
        assert check_winner(board) == "O"
    
    def test_no_winner_incomplete_board(self):
        board = ["X", "X", "", "O", "O", "", "", "", ""]
        assert check_winner(board) is None
    
    def test_no_winner_mixed_board(self):
        board = ["X", "O", "X", "O", "X", "O", "O", "X", ""]
        assert check_winner(board) is None


class TestIsBoardFull:
    """Test is_board_full function."""
    
    def test_empty_board_not_full(self):
        board = ["" for _ in range(9)]
        assert is_board_full(board) is False
    
    def test_board_with_one_empty_not_full(self):
        board = ["X", "O", "X", "O", "X", "O", "O", "X", ""]
        assert is_board_full(board) is False
    
    def test_full_board(self):
        board = ["X", "O", "X", "O", "X", "O", "O", "X", "X"]
        assert is_board_full(board) is True
    
    def test_full_board_all_x(self):
        board = ["X"] * 9
        assert is_board_full(board) is True


class TestGetAvailableMoves:
    """Test get_available_moves function."""
    
    def test_empty_board_all_moves_available(self):
        board = ["" for _ in range(9)]
        available = get_available_moves(board)
        assert available == list(range(9))
    
    def test_no_moves_available(self):
        board = ["X", "O", "X", "O", "X", "O", "O", "X", "X"]
        available = get_available_moves(board)
        assert available == []
    
    def test_some_moves_available(self):
        board = ["X", "", "O", "", "", "", "", "", ""]
        available = get_available_moves(board)
        assert 1 in available
        assert 3 in available
        assert 4 in available
        assert 0 not in available
        assert 2 not in available
    
    def test_moves_are_sorted(self):
        board = ["X", "", "O", "", "X", "", "", "", ""]
        available = get_available_moves(board)
        assert available == sorted(available)


class TestMinimax:
    """Test minimax algorithm."""
    
    def test_minimax_ai_winning_position(self):
        board = ["O", "O", "", "X", "X", "", "", "", ""]
        score = minimax(board, 0, True)
        assert score == 1
    
    def test_minimax_human_winning_position(self):
        board = ["X", "X", "", "O", "O", "", "", "", ""]
        score = minimax(board, 0, False)
        assert score == -1
    
    def test_minimax_draw_position(self):
        board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        score = minimax(board, 0, True)
        assert score == 0
    
    def test_minimax_empty_board_maximizing(self):
        board = ["" for _ in range(9)]
        score = minimax(board, 0, True)
        assert score in [-1, 0, 1]
    
    def test_minimax_empty_board_minimizing(self):
        board = ["" for _ in range(9)]
        score