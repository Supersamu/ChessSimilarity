"""
parse_pgn.py
Given a pgn, creates a list of tensors for training / validation.
"""

import chess
import chess.pgn
import numpy as np
import torch
import json
import io
from lichess_data_loading.gm_usernames import Lichess_names



def pgn_to_8x8_arrays(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    board_states = []

    # Map pieces to unique integers
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6  # Black pieces
    }

    # Iterate through moves
    for move in game.mainline_moves():
        board.push(move)
        # Create an 8x8 array for the current board state
        board_array = np.zeros((8, 8), dtype=int)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                board_array[row, col] = piece_map[piece.symbol()]
        board_states.append(board_array)

    return board_states


if __name__ == "__main__":

    with open('src/lichess_data_loading/gm_games.json') as f:
        data = json.load(f)

    example_pgn = data["white"][Lichess_names[0]][0]  # Example PGN from the first GM's white games
    print(example_pgn)
    arrays = pgn_to_8x8_arrays(example_pgn)
    print(arrays)

    # Convert to PyTorch tensors if needed
    tensor_list = [torch.tensor(arr, dtype=torch.float32) for arr in arrays]
