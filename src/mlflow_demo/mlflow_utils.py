from torch.utils.data import DataLoader
from typing import List, Tuple
from model_utils.data_loader import create_dataloaders


def create_train_val_test_split(player_names: List[str], batch_size: int, train_val_vs_test_split: float, 
                                train_vs_val_split: float, random_seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train, validation, and test DataLoaders.

    player_names: List[str]: List of player names.
    batch_size: int: Batch size for DataLoaders.
    train_val_vs_test_split: float: Split ratio for train/val vs test.
    train_vs_val_split: float: Split ratio for train vs validation.
    random_seed: int: Random seed for reproducibility.

    Returns:
    Tuple[DataLoader, DataLoader, DataLoader, int]: Train, validation, and test DataLoaders, and number of classes.
    """
    _, test_loader, _, train_and_val_indices, test_indices = create_dataloaders(player_names=player_names, batch_size=batch_size, 
                                                             train_split=train_val_vs_test_split, random_seed=random_seed)
    train_loader, val_loader, num_classes, train_indices, val_indices = create_dataloaders(player_names=player_names, available_indices=train_and_val_indices, 
                                                                     batch_size=batch_size, train_split=train_vs_val_split, random_seed=random_seed)
    return train_loader, val_loader, test_loader, num_classes