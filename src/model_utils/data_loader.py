import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class ChessDataset(Dataset):
    """
    Custom Dataset class for chess similarity data.
    Loads samples and labels from .pt files.
    """
    
    def __init__(self, samples_path: str, labels_path: str):
        """
        Initialize the dataset.
        
        Args:
            samples_path: Path to samples.pt file
            labels_path: Path to labels.pt file
        """
        try:
            self.samples = torch.load(samples_path, map_location='cpu')
            self.labels = torch.load(labels_path, map_location='cpu')
            
            # Ensure samples and labels have the same length
            assert len(self.samples) == len(self.labels), \
                f"Samples ({len(self.samples)}) and labels ({len(self.labels)}) must have the same length"
                
            print(f"Loaded dataset: {len(self.samples)} samples")
            print(f"Sample shape: {self.samples.shape}")
            print(f"Label shape: {self.labels.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


def create_dataloaders(samples_path: str = 'src/data/samples.pt', 
                      labels_path: str = 'src/data/labels.pt',
                      batch_size: int = 32,
                      train_split: float = 0.8,
                      random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        samples_path: Path to samples file
        labels_path: Path to labels file
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    
    # Create and split dataset
    dataset = ChessDataset(samples_path, labels_path)
    
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
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
    
    return train_loader, val_loader