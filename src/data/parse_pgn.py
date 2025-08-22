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



def pgn_to_3d_arrays(pgn):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    board_states = []

    # Map pieces to unique integers
    piece_map = {
        'P': 5, 'N': 4, 'B': 3, 'R': 2, 'Q': 1, 'K': 0,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
    }
    # For saving the boardstates as tensors, we are now able to flip
    # the tensor along the first dimension to get the game from the the other direction
    
    # Iterate through moves
    for move in game.mainline_moves():
        board.push(move)
        # Create an 8x8 array for the current board state
        board_array = np.zeros((12, 8, 8), dtype=int)  # channel-first
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                board_array[piece_map[piece.symbol()], row, col] = 1
        board_states.append(board_array)

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


if __name__ == "__main__":
    with open('src/lichess_data_loading/gm_games.json') as f:
        data = json.load(f)
    list_of_labels = []
    list_of_samples = []
    for label, lichess_name in enumerate(Lichess_names):
        for color in ["white", "black"]:
            pgns_as_color = data[color][lichess_name]
            for pgn in pgns_as_color:
                boardstates = pgn_to_3d_arrays(pgn)
                samples = transform_board_arrays_to_samples(color, boardstates)
                list_of_samples.extend(samples)
                list_of_labels.extend([label] * len(samples))
    # Convert the samples and labels to tensors and save them
    x = np.array(list_of_samples, dtype=np.float32)
    y = np.array(list_of_labels, dtype=np.int64)
    torch.save(torch.tensor(np.array(list_of_samples), dtype=torch.float32), 'src/data/samples.pt')
    torch.save(torch.tensor(np.array(list_of_labels), dtype=torch.int64), 'src/data/labels.pt')

    