import torch
import torch.nn as nn
from functools import reduce


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


def create_model(input_tensor, output_tensor):
    """
    Create a chess similarity model.

    Args:
        input_tensor: Input tensor
        output_tensor: Output tensor

    Returns:
        model: The created model
    """
    # the input is a 5D tensor with shape (batch_size, channels, time, height, width)
    input_size = reduce(lambda x, y: x * y, input_tensor.shape[1:])  # channels*time*height*width
    print(f"Input size: {input_size}")
    num_classes = len(torch.unique(output_tensor))
    
    # Create model
    model = ChessNN(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        num_classes=num_classes,
        dropout_rate=0.2
    )
    return model