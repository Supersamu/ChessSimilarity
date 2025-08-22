from data_loader import create_dataloaders
from simple_test_model import ChessNN
from trainer import ChessModelTrainer
import torch
from functools import reduce


def main():
    """
    Main function to demonstrate the model training process.
    """
    print("Chess Similarity Model Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        samples_path='src/data/samples.pt',
        labels_path='src/data/labels.pt',
        batch_size=32
    )
    
    # Get sample batch to determine input size
    sample_batch_features, sample_batch_labels = next(iter(train_loader))
    # the input is a 5D tensor with shape (batch_size, channels, time, height, width)
    input_size = reduce(lambda x, y: x * y, sample_batch_features.shape[1:])  # channels*time*height*width
    print(f"Input size: {input_size}")
    num_classes = len(torch.unique(sample_batch_labels))
    
    # Create model
    model = ChessNN(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        num_classes=num_classes,
        dropout_rate=0.2
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = ChessModelTrainer(model, device)
    
    # Train model
    trainer.train(train_loader, val_loader, epochs=10)
    
    # Save model
    torch.save(model.state_dict(), 'chess_similarity_model.pth')
    print("\nModel saved as 'chess_similarity_model.pth'")


if __name__ == "__main__":
    main()