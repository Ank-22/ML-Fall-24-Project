import torch
from torch import nn, optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Loader class
class DataLoaderPreprocessor:
    def __init__(self, csv_path, target_col, test_size=0.2, batch_size=32, window_size=50):
        self.csv_path = csv_path
        self.target_col = target_col
        self.test_size = test_size
        self.batch_size = batch_size
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

    def test_load(self):
        data = pd.read_csv(self.csv_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.sort_values(by='Datetime', inplace=True)  # Ensure data is sorted by datetime
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        feature_columns = [col for col in data.columns if col not in [self.target_col, 'Datetime', 'Delta_Target']]
        X = data[feature_columns].values
        y = data[self.target_col].values

        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
        X, y = self.create_sequences(X, y, self.window_size)
        test_data = torch.utils.data.TensorDataset(X, y)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
  
        return test_loader, X.shape[2], data["Datetime"][50:]  # Return the number of features as input dimension


    def load_and_preprocess(self):
        """Load and preprocess data."""
        data = pd.read_csv(self.csv_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.sort_values(by='Datetime', inplace=True)  # Ensure data is sorted by datetime
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        feature_columns = [col for col in data.columns if col not in [self.target_col, 'Datetime', 'Delta_Target']]
        X = data[feature_columns].values
        y = data[self.target_col].values

        # Scale features
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)

        # Create sequences
        X, y = self.create_sequences(X, y, self.window_size)

        # Split into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
   
        return train_loader, test_loader, X.shape[2]  # Return the number of features as input dimension


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))

class GRUTrainer:
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, lr=0.001, epochs=50, device=None, save_path="best_model.pth"):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = 1 
        self.save_path = save_path
        self.best_loss = float('inf')  # Initialize with infinity for tracking the best model
        self.model = self.GRUModel(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers).to(self.device)
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

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

    def load(self, path):
        """Load the model from a file."""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def save(self):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def train(self, train_loader, threshold=1e-4, patience=5):
        """Train the GRU model with WandB logging."""
        self.model.train()
        no_improvement_epochs = 0  # Counter for epochs without significant loss improvement
        previous_loss = float('inf')
        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                

                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                #print("loss : ", loss)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Calculate average loss and log it to WandB
            avg_loss = epoch_loss / len(train_loader)
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

            # Check loss improvement
            loss_change = abs(previous_loss - avg_loss)
            if loss_change < threshold:
                no_improvement_epochs += 1
            else:
                no_improvement_epochs = 0  # Reset counter if improvement is significant

            previous_loss = avg_loss

            # Adjust learning rate if no significant improvement
            if no_improvement_epochs >= patience:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5  # Reduce learning rate by half
                print(f"Reducing learning rate to {self.optimizer.param_groups[0]['lr']:.6f}")
                no_improvement_epochs = 0  # Reset counter after adjustment

            # Save the best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save()

    def evaluate(self, test_loader, data_datetime=None):
        """Evaluate the GRU model on the test data and log predictions to WandB."""
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())

        # Print test loss of the model
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        test_loss = self.criterion(torch.tensor(predictions), torch.tensor(actuals)).item()
        
        # Calculate and log performance metrics
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        ss_total = np.sum((actuals - np.mean(actuals)) ** 2)
        ss_residual = np.sum((actuals - predictions) ** 2)
        r2_score = 1 - (ss_residual / ss_total)
        correct_directions = np.sum(
            np.sign(predictions[1:] - predictions[:-1]) == np.sign(actuals[1:] - actuals[:-1])
        )
        directional_accuracy = correct_directions / (len(predictions) - 1) * 100

        wandb.log({
            "Test_Loss": test_loss,
            "MAE": mae,
            "MAPE": mape,
            "R2_Score": r2_score,
            "Directional_Accuracy": directional_accuracy,
        })

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"R² Score: {r2_score:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")

        # Log predictions vs actuals to WandB
        wandb.log({"predictions_vs_actuals": wandb.Table(data=[[p, a] for p, a in zip(predictions, actuals)], 
                                                        columns=["Prediction", "Actual"])})
        
        # Save predictions to a CSV file
        if data_datetime is not None:
            self.save_predictions(data_datetime, predictions, actuals)
            self.plot_predictions() # Plot the predictions vs actuals

        return predictions, actuals

    def save_predictions(self, data_datetime, predictions, actuals):
        """Save the predictions and actuals to a CSV file."""
        predictions = [p[0] if isinstance(p, (list, np.ndarray)) else p for p in predictions]
        actuals = [a[0] if isinstance(a, (list, np.ndarray)) else a for a in actuals]

        df = pd.DataFrame({"DateTime": data_datetime,"predictions": predictions, "actuals": actuals})
        df.to_csv("predictions_vs_actuals.csv", index=False)

    #Function to read the csv file and plot the predictions vs actuals
    def plot_predictions(self):
        df = pd.read_csv('predictions_vs_actuals.csv')
        ax = df.plot(x="DateTime", y=["predictions", "actuals"], title="Predictions vs Actuals")
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Values")
        plt.show()

    def run(self, train_loader, test_loader, wandb_config=None):
        """End-to-end pipeline for training and evaluating the model."""

        print("Training model...")
        self.train(train_loader)

        print("Evaluating model...")
        predictions, actuals = self.evaluate(test_loader)
        for i in range(10):  # Print the first 10 results
            print(f"Predicted: {predictions[i][0]:.4f}, Actual: {actuals[i][0]:.4f}")

        if wandb_config:
            wandb.finish()
