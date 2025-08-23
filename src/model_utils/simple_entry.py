import torch
from functools import reduce
from data_loader import create_dataloaders
from simple_test_model import create_model as create_simple_model
from base_model import create_model as create_base_model
from trainer import ChessModelTrainer
from lichess_data_loading.gm_usernames import Lichess_names

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
    train_loader, val_loader, num_classes, _, _ = create_dataloaders(
        player_names=Lichess_names,
    )
    # Get sample batch to determine input size
    sample_batch_features, _ = next(iter(train_loader))
    # the input is a 5D tensor with shape (batch_size, channels, depth, height, width)
    model = create_base_model(sample_batch_features, num_classes)
    
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