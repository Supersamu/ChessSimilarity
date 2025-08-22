import torch
import torch.nn as nn


class ChessNN(nn.Module):
    """
    Neural Network for chess similarity prediction.
    """
    
    def __init__(self, input_size: int, hidden_sizes: list = [128, 64, 32], 
                 num_classes: int = 2, dropout_rate: float = 0.2):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(ChessNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(start_dim=1))


