import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from functools import reduce
from collections import defaultdict


class ChessDataset(Dataset):
    """
    Custom Dataset class for chess similarity data.
    Loads samples and labels from .pt files.
    """

    def __init__(self, player_names: List[str], indices: List[Tuple[List[int]]] = None):
        """
        Initialize the dataset.
        """
        self.player_names = player_names
        self.indices = indices
        self.features = []
        self.labels = []
        for player_name, indices_for_player in zip(player_names, indices):
            try:
                for color, indices_for_player_by_color in zip(["white", "black"], indices_for_player):
                    samples_for_color_and_player = [
                        torch.load(f'src/data/data_per_player/{player_name}/features_{color}_{i}.pt', map_location='cpu') 
                        for i in indices_for_player_by_color]
                    labels_for_color_and_player = [
                        torch.load(f'src/data/data_per_player/{player_name}/labels_{color}_{i}.pt', map_location='cpu') 
                        for i in indices_for_player_by_color]
                    self.features.extend(samples_for_color_and_player)
                    self.labels.extend(labels_for_color_and_player)
            except Exception as e:
                print(f"Error loading data for {player_name}: {e}")

        # Concatenate all samples and labels
        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        # Ensure samples and labels have the same length
        assert len(self.features) == len(self.labels), \
            f"Samples ({len(self.features)}) and labels ({len(self.labels)}) must have the same length"

        print(f"Loaded dataset: {len(self.features)} samples")
        print(f"Sample shape: {self.features.shape}")
        print(f"Label shape: {self.labels.shape}")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_dataloaders(player_names: List[str],
                       available_indices: List[Tuple[List[int]]] = None,
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, 
                                                       List[Tuple[List[int]]], List[Tuple[List[int]]]]:
    """
    Create train and validation dataloaders.
    
    Args:
        player_names: List of player names
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    np.random.seed(random_seed)

    # load information about player data
    train_indices = []
    val_indices = []
    running_sum_train_games = 0
    running_sum_train_moves = 0
    running_sum_val_games = 0
    running_sum_val_moves = 0
    for player_idx, player_name in enumerate(player_names):
        train_indices_by_color = defaultdict(list)
        val_indices_by_color = defaultdict(list)
        for color_idx, color in enumerate(["white", "black"]):
            with open(f'src/data/data_per_player/{player_name}/info_{color}.txt', 'rb') as f:
                num_moves_per_game = [int(line) for line in f.readlines()]
            print(len(num_moves_per_game), f"games found for {player_name} as {color}")
            # shuffle games and find a set of indices such that the sum of moves approximates
            # the percentage of available allocated moves
            if not available_indices:
                total_moves = sum(num_moves_per_game)
                shuffled_indices = np.random.permutation(np.arange(len(num_moves_per_game)))
            else:
                total_moves = sum([num_moves_per_game[i] for i in available_indices[player_idx][color_idx]])
                shuffled_indices = np.random.permutation(available_indices[player_idx][color_idx])
            
            train_moves = int(total_moves * train_split)

            current_sum = 0
            for index in shuffled_indices:
                num_moves = num_moves_per_game[index]
                if abs(train_moves - (current_sum + num_moves)) > abs(train_moves - current_sum):
                    # If adding this game overshoots the target by more than the target was undershot, stop adding
                    break
                current_sum += num_moves
                train_indices_by_color[color].append(index)
            
            running_sum_train_moves += current_sum
            running_sum_val_moves += total_moves - current_sum
            # put remaining indices into val_indices
            val_indices_by_color[color] = [i for i in shuffled_indices if i not in train_indices_by_color[color]]
            running_sum_train_games += len(train_indices_by_color[color])
            running_sum_val_games += len(shuffled_indices) - len(train_indices_by_color[color])
        train_indices.append((train_indices_by_color["white"], train_indices_by_color["black"]))
        val_indices.append((val_indices_by_color["white"], val_indices_by_color["black"]))
    # report statistics of data split
    print(f"{running_sum_val_games + running_sum_train_games} Available games were split into {running_sum_train_games} " + 
          f"training and {running_sum_val_games} validation games, a percentage of {running_sum_train_games / (running_sum_train_games + running_sum_val_games) * 100}% training games")
    print(f"{running_sum_val_moves + running_sum_train_moves} Available moves were split into {running_sum_train_moves} " + 
          f"training and {running_sum_val_moves} validation moves, a percentage of {running_sum_train_moves / (running_sum_train_moves + running_sum_val_moves) * 100}% training moves")
    
    num_classes = len(player_names)
    train_dataset = ChessDataset(player_names=player_names, indices=train_indices)
    val_dataset = ChessDataset(player_names=player_names, indices=val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        generator=generator
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    return train_loader, val_loader, num_classes, train_indices, val_indices