import torch
from torch import nn, optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Data Loader class
class DataLoaderPreprocessor:
    def __init__(self, csv_path, target_col, test_size=0.2, batch_size=32, random_state=42, window_size=50):
        self.csv_path = csv_path
        self.target_col = target_col
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.window_size = window_size
        self.scaler = None

    def create_sequences(self, data, target, window_size):
        """Create input sequences and corresponding target values."""
        sequences = []
        targets = []
        for i in range(len(data) - window_size):
            seq = data[i : i + window_size]
            label = target[i + window_size]
            sequences.append(seq)
            targets.append(label)
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def load_and_preprocess(self):
        """Load and preprocess data."""
        data = pd.read_csv(self.csv_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.sort_values(by='Datetime', inplace=True)  # Ensure data is sorted by datetime
        data.ffill(inplace=True)

        feature_columns = [col for col in data.columns if col not in [self.target_col, 'Datetime', 'Delta_Target']]
        X = data[feature_columns].values
        y = data[self.target_col].values

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Create sequences
        X, y = self.create_sequences(X, y, self.window_size)
        #print("X shape : ", X.shape)
        #print("Y shape : ", y.shape)

        # Split into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, X.shape[2]  # Return the number of features as input dimension



class GRUTrainer:
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, lr=0.001, epochs=50, device=None, save_path="best_model.pth"):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.input_dim = input_dim
        self.output_dim = 1  # Assuming regression
        self.save_path = save_path
        self.best_loss = float('inf')  # Initialize with infinity for tracking the best model

    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super(GRUTrainer.GRUModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out

    def initialize_model(self):
        """Initialize the GRU model, loss function, and optimizer."""
        self.model = self.GRUModel(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def save(self):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def train(self, train_loader):
        """Train the GRU model with WandB logging."""
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                

                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                print("loss : ", loss)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Calculate average loss and log it to WandB
            print("train_loader len : ", len(train_loader))
            print("epoch_loss : ", epoch_loss)
            print("loss.item() : ", loss.item())
            avg_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

            # Save the best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save()

    def evaluate(self, test_loader):
        """Evaluate the GRU model on the test data and log predictions to WandB."""
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())

        # Log predictions vs actuals to WandB
        wandb.log({"predictions_vs_actuals": wandb.Table(data=[[p, a] for p, a in zip(predictions, actuals)], 
                                                        columns=["Prediction", "Actual"])})
        return predictions, actuals

    def run(self, train_loader, test_loader, wandb_config=None):
        """End-to-end pipeline for training and evaluating the model."""
        if wandb_config:
            wandb.init(
                project=wandb_config.get("project", "gru-project"),
                name=wandb_config.get("run_name", "gru-run"),
                config=wandb_config
            )

        print("Initializing model...")
        self.initialize_model()

        print("Training model...")
        self.train(train_loader)

        print("Evaluating model...")
        predictions, actuals = self.evaluate(test_loader)
        for i in range(10):  # Print the first 10 results
            print(f"Predicted: {predictions[i][0]:.4f}, Actual: {actuals[i][0]:.4f}")

        if wandb_config:
            wandb.finish()
