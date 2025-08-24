import mlflow
from mlflow.tracking import MlflowClient
from mlflow_utils import create_train_val_test_split
from lichess_data_loading.gm_usernames import Lichess_names
from model_utils.trainer import ChessModelTrainer
# Initialize MLflow client
client = MlflowClient()

all_experiments = client.search_experiments()

model = mlflow.pytorch.load_model("models:/current_best@champion")
model_version = client.get_model_version_by_alias("current_best", "champion")

# run inference with model
train_loader, val_loader, test_loader, num_classes = create_train_val_test_split(player_names=Lichess_names,
                                                                       batch_size=int(model_version.params["batch_size"]),
                                                                       train_val_vs_test_split=float(model_version.params["train_val_vs_test_split"]),
                                                                       train_vs_val_split=float(model_version.params["train_vs_val_split"]),
                                                                       random_seed=int(model_version.params["random_seed"]))
# calculate train, val and test metrics
trainer = ChessModelTrainer(model, lr=0.001, optimizer_name="Adam", device="cpu")
for data_type, dataloader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
    loss, accuracy = trainer.evaluate(dataloader)
    print(f"{data_type} Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
