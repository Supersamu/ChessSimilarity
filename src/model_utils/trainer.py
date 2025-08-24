import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional

class ChessModelTrainer:
    """
    Trainer class for the chess similarity model.
    """
    
    def __init__(self, model: nn.Module, lr: float, optimizer_name: str = 'adam', device: str = 'cpu'):
        """
        Initialize the trainer.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def run_epoch(self, dataloader, training: bool):
        total_loss = 0.0
        correct = 0
        total = 0

        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            if training:
                self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            if training:
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        
        accuracy, avg_loss = self.run_epoch(dataloader, training=True)

        return avg_loss, accuracy

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model and return loss and accuracy."""
        self.model.eval()
        
        with torch.no_grad():
            avg_loss, accuracy = self.run_epoch(dataloader, training=False)
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
              epochs: int = 10):
        """Train the model for multiple epochs."""
        print(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

