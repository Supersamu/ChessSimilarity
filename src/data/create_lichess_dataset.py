import json
import torch
import numpy as np
import os
from lichess_data_loading.gm_usernames import Lichess_names
from parse_pgn import pgn_to_3d_arrays, transform_board_arrays_to_samples


def create_player_dataset(lichess_name, full_data):
    """
    Create a dataset for a specific player, with each game being saved in a separate file.
    In the player directory, create a file showing the number of samples per file.

    Args:
        lichess_name: The Lichess username of the player.
        color: The color of the pieces (white or black).
        pgns: A list of PGN strings for the player's games.

    Returns:
        A tuple (samples, labels) where samples is a list of input tensors
        and labels is a list of corresponding output tensors.
    """
    
    os.makedirs(f'src/data/data_per_player/{lichess_name}', exist_ok=True)
    label = Lichess_names.index(lichess_name)
    print(label)
    for color in ["white", "black"]:
        # initialize set of pgns to exclude duplicates
        seen_pgns = set()
        duplicate_count = 0
        with open(f'src/data/data_per_player/{lichess_name}/info_{color}.txt', 'w') as f:
            pgns_as_color = full_data[color][lichess_name]
            for idx, pgn in enumerate(pgns_as_color):
                if pgn in seen_pgns:
                    print(f"Duplicate game found for {lichess_name} as {color}, skipping.")
                    duplicate_count += 1
                    continue
                seen_pgns.add(pgn)
                boardstates = pgn_to_3d_arrays(pgn)
                samples = transform_board_arrays_to_samples(color, boardstates)
                labels = [label] * len(samples)
                f.write(f"{len(samples)}\n")
                torch.save(torch.tensor(np.array(samples), dtype=torch.float32), f'src/data/data_per_player/{lichess_name}/features_{color}_{idx-duplicate_count}.pt')
                torch.save(torch.tensor(np.array(labels), dtype=torch.int64), f'src/data/data_per_player/{lichess_name}/labels_{color}_{idx-duplicate_count}.pt')
    return samples, labels


if __name__ == "__main__":
    with open('src/lichess_data_loading/gm_games.json') as f:
        data = json.load(f)
    for lichess_name in Lichess_names:
        print(f"Creating dataset for {lichess_name}")
        create_player_dataset(lichess_name, data)