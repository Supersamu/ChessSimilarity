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

def board_to_3d_array(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board to a 3D numpy array.
    """
    board_array = np.zeros((12, 8, 8), dtype=int)  # channel-first
    piece_map = {
        'P': 5, 'N': 4, 'B': 3, 'R': 2, 'Q': 1, 'K': 0,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            board_array[piece_map[piece.symbol()], row, col] = 1
    return board_array


def pgn_to_3d_arrays(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    board_states = [board_to_3d_array(board)]
    
    # Iterate through moves
    for move in game.mainline_moves():
        board.push(move)
        board_states.append(board_to_3d_array(board))

    return board_states


def transform_board_arrays_to_samples(color, boardstates):
    starting_idx = {"white": 0, "black": 1}[color]
    samples = []
    for original_position, subsequent_position in zip(boardstates[starting_idx::2], boardstates[starting_idx + 1::2]):
        # if the player of interest is black, we need to flip along the first axis 
        # because we always want to look at the board from the same direction
        if color == "black":
            original_position = np.flip(original_position, axis=0)
            subsequent_position = np.flip(subsequent_position, axis=0)
        samples.append(np.stack((original_position, subsequent_position)))  # shape: 2, 12, 8, 8
    return samples

