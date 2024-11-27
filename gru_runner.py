import argparse
import yaml
import torch
from gru_trainer import DataLoaderPreprocessor, GRUTrainer

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run GRU training or testing.")
    
    # Configuration file
    parser.add_argument("--config", type=str, default="gru_config.yaml", help="Path to the config file.")
    
    # Command-line overrides for key parameters
    parser.add_argument("--csv_path", type=str, help="Path to the dataset CSV file.")
    parser.add_argument("--target_col", type=str, help="Target column in the dataset.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training/testing.")
    parser.add_argument("--lr", type=float, help="Learning rate for training.")
    parser.add_argument("--epochs", type=int, help="Number of epochs for training.")
    parser.add_argument("--save_path", type=str, help="Path to save the best model.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Mode to run: train or test.")
    parser.add_argument("--wandb_enable", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--wandb_project", type=str, help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, help="WandB run name.")

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override configuration with command-line arguments if provided
    if args.csv_path:
        config["data"]["csv_path"] = args.csv_path
    if args.target_col:
        config["data"]["target_col"] = args.target_col
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.save_path:
        config["training"]["save_path"] = args.save_path
    if args.wandb_enable is not None:
        config["wandb"]["enable"] = args.wandb_enable
    if args.wandb_project:
        config["wandb"]["project"] = args.wandb_project
    if args.wandb_run_name:
        config["wandb"]["run_name"] = args.wandb_run_name

    # Device setup
    device = torch.device("cuda" if config["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading data...")
    data_loader = DataLoaderPreprocessor(
        csv_path=config["data"]["csv_path"],
        target_col=config["data"]["target_col"],
        test_size=config["data"]["test_size"],
        batch_size=config["data"]["batch_size"],
        random_state=config["data"]["random_state"]
    )
    train_loader, test_loader, input_dim = data_loader.load_and_preprocess()

    # Initialize the GRU trainer
    print("Initializing GRU model...")
    trainer = GRUTrainer(
        input_dim=input_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        lr=config["training"]["lr"],
        epochs=config["training"]["epochs"],
        device=device,
        save_path=config["training"]["save_path"]
    )

    # Set WandB configuration if enabled
    wandb_config = None
    if config["wandb"]["enable"]:
        wandb_config = {
            "project": config["wandb"]["project"],
            "run_name": config["wandb"]["run_name"],
            "epochs": config["training"]["epochs"],
            "hidden_dim": config["model"]["hidden_dim"],
            "learning_rate": config["training"]["lr"]
        }

    # Run training or testing
    if args.mode == "train":
        print("Training the model...")
        trainer.run(train_loader, test_loader, wandb_config=wandb_config)
    elif args.mode == "test":
        print("Testing the model...")
        predictions, actuals = trainer.evaluate(test_loader)
        for i in range(10):  # Print first 10 results
            print(f"Predicted: {predictions[i][0]:.4f}, Actual: {actuals[i][0]:.4f}")

if __name__ == "__main__":
    main()
