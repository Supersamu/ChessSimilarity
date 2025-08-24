import mlflow
import optuna
from functools import partial
from mlflow_utils import create_train_val_test_split
from model_utils.base_model import ChessCNN
from model_utils.trainer import ChessModelTrainer
from lichess_data_loading.gm_usernames import Lichess_names


def objective(trial: optuna.Trial) -> float:
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    num_channels = [trial.suggest_int(f"n_channels_l{i}", 16, 128, step=16) for i in range(n_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    channels_dict = {f"layer_{i}": num_channels[i] for i in range(n_layers)}
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    with mlflow.start_run(nested=True):
        # Log hyperparameters
        params = {
            "lr": lr,
            "optimizer": optimizer_name,
            "batch_size": batch_size,
            "epochs": 15,
            "dropout_rate": dropout_rate,
            "random_seed": 42,
            "train_val_vs_test_split": 0.8,
            "train_vs_val_split": 0.8,
            "device": "cpu",
            **channels_dict
        }
        mlflow.log_params(params)
        train_loader, val_loader, _, num_classes = create_train_val_test_split(player_names=Lichess_names, 
                                                                               batch_size=params["batch_size"],
                                                                               train_val_vs_test_split=params["train_val_vs_test_split"],
                                                                               train_vs_val_split=params["train_vs_val_split"],
                                                                               random_seed=params["random_seed"])
        # Create model
        model = ChessCNN(input_channels=12, hidden_channels=num_channels, 
                         dropout_rate=params["dropout_rate"], num_classes=num_classes)
        trainer = ChessModelTrainer(model, lr=params["lr"], optimizer_name=params["optimizer"], 
                                    device=params["device"])

        # Train for a few epochs
        best_val_acc = 0
        for epoch in range(params["epochs"]):
            train_loss, train_acc = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.evaluate(val_loader)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            best_val_acc = max(best_val_acc, val_acc)

        # Final logging
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.pytorch.log_model(model, name="model")

        return best_val_acc


# Execute hyperparameter search
with mlflow.start_run(run_name="hyperparam_optimization"):
    study = optuna.create_study(direction="maximize")
    objective_func = partial(
        objective, 
    )
    study.optimize(objective_func, n_trials=20)

    # Log best parameters and score
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_val_accuracy", study.best_value)