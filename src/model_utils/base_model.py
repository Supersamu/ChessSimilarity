import torch
import torch.nn as nn


class ChessCNN(nn.Module):
    """
    Neural Network for chess similarity prediction.
    """
    
    def __init__(self, input_channels: int, hidden_channels: list = [128, 64, 32], 
                 num_classes: int = 2, dropout_rate: float = 0.2):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(ChessCNN, self).__init__()

        cnn_layers = []
        prev_channel = input_channels
        for hidden_channel in hidden_channels[:-1]:
            cnn_layers.append(nn.Conv2d(prev_channel, hidden_channel, kernel_size=3, padding=1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.BatchNorm2d(hidden_channel))
            cnn_layers.append(nn.Dropout(dropout_rate))
            prev_channel = hidden_channel

        self.first_network = nn.Sequential(*cnn_layers)

        self.second_network = nn.Sequential(
            nn.Conv2d(prev_channel, hidden_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels[-1]),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(hidden_channels[-1] * 8 * 8, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # split data into two streams (original and subsequent positions)
        x1, x2 = x.chunk(2, dim=1)
        x1, x2 = x1.squeeze(1), x2.squeeze(1)
        out1 = self.first_network(x1)
        out2 = self.first_network(x2)
        # merge the streams
        x = out1 + out2
        return self.second_network(x)




def create_model(input_tensor: torch.Tensor, num_classes: int):
    """
    Create a chess similarity model.

    Args:
        input_tensor: Input tensor
        num_classes: Number of output classes

    Returns:
        model: The created model
    """
    # the input is a 5D tensor with shape (batch_size, channels, time, height, width)
    num_channels = input_tensor.shape[2]
    # Create model
    model = ChessCNN(
        input_channels=num_channels,
        hidden_channels=[16, 16, 16],
        num_classes=num_classes,
        dropout_rate=0.2
    )
    return model