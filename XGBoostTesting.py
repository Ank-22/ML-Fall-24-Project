import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

# Constants
TRAIN_FILE_PATHS = [
    'Dataset/INFY_DATA.csv', 
    'Dataset/LTIM_DATA.csv', 
    'Dataset/TCS_DATA.csv', 
    'Dataset/WIPRO_DATA.csv', 
    'Dataset/PERSISTENT_DATA.csv'
]
TEST_FILE_PATHS = [
    'Test_Dataset/INFY_DATA_TEST.csv',
    'Test_Dataset/LTIM_DATA_TEST.csv',
    'Test_Dataset/TCS_DATA_TEST.csv',
    'Test_Dataset/WIPRO_DATA_TEST.csv',
    'Test_Dataset/PERSISTENT_DATA_TEST.csv'
]

COLUMNS_TO_FILL = ['Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20', 
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']
FEATURE_COLUMNS = ['Open', 'Volume', 'PE_Ratio', '52_Week_High', '52_Week_Low',
                   'Is_52_week_high', 'Is_52_week_low', 'Is_high', 'Is_low',
                   'Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20',
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']

# Hyperparameters for XGBoost
HYPERPARAMETERS = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.06,
    'max_depth': 5,
    'n_estimators': 240,
    'subsample': 0.8
}

def load_and_preprocess_data(file_path, scaler=None):
    # Load data
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['hour'] = data['Datetime'].dt.hour
    data['minute'] = data['Datetime'].dt.minute
    data['dayofweek'] = data['Datetime'].dt.dayofweek

    # Fill missing values
    data[COLUMNS_TO_FILL] = data[COLUMNS_TO_FILL].fillna(method='ffill')

    # Scale features
    if scaler is None:
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(data[FEATURE_COLUMNS])
    else:
        features_scaled = scaler.transform(data[FEATURE_COLUMNS])

    features = pd.DataFrame(features_scaled, columns=FEATURE_COLUMNS)
    target = data['Target']

    return features, target, scaler

def train_model(X_train, y_train, **kwargs):
    """Train an XGBoost model with predefined hyperparameters."""
    model = XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        **kwargs
    )
    # Track training loss
    eval_set = [(X_train, y_train)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return RMSE and predictions."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    return y_pred, rmse

def plot_rmse_comparison(comparison_results, filepath):
    """Plot RMSE comparison for training and testing datasets."""
    results_df = pd.DataFrame(comparison_results)
    datasets = results_df["Dataset"]
    train_rmse = results_df["Train_RMSE"]
    test_rmse = results_df["Test_RMSE"]

    x = np.arange(len(datasets))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_rmse, width=width, label='Train RMSE', color='skyblue', alpha=0.8)
    plt.bar(x + width/2, test_rmse, width=width, label='Test RMSE', color='orange', alpha=0.8)

    plt.xlabel("Dataset")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison: Train vs Test")
    plt.xticks(x, datasets, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"RMSE comparison plot saved to {filepath}")

def plot_training_losses(model, dataset_name, output_dir):
    """
    Plot training losses over iterations for the given model.
    """
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['rmse'], label="Train Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title(f"Training Losses: {dataset_name}")
    plt.grid(True)
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"{dataset_name}_training_losses_only.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Training loss plot saved to {filepath}")

def plot_actual_vs_predicted(y_true, y_pred, dataset_name, output_dir):
    """
    Plot actual vs. predicted values for the given dataset.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label="Actual", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Actual vs Predicted: {dataset_name}")
    plt.grid(True)
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"{dataset_name}_actual_vs_predicted.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Actual vs Predicted plot saved to {filepath}")

def save_predictions_to_csv(y_true, y_pred, test_file, dataset_name, output_dir):
    """
    Save true and predicted values to a CSV file, including Datetime.
    """
    # Load the test file to get the Datetime column
    test_data = pd.read_csv(test_file)
    
    # Ensure the Datetime column exists
    if 'Datetime' in test_data.columns:
        datetimes = pd.to_datetime(test_data['Datetime'])
    else:
        raise ValueError("The test dataset does not have a 'Datetime' column.")
    
    # Create the DataFrame
    predictions_df = pd.DataFrame({
        "Datetime": datetimes,
        "True Values": y_true.values,
        "Predicted Values": y_pred
    })
    
    # Save to CSV
    filepath = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
    predictions_df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")


def main():
    output_dir = "results_final_testing"
    os.makedirs(output_dir, exist_ok=True)

    comparison_results = {"Dataset": [], "Train_RMSE": [], "Test_RMSE": []}

    for train_file, test_file in zip(TRAIN_FILE_PATHS, TEST_FILE_PATHS):
        print(f"Processing {train_file} with test file {test_file}...")

        # Load training data
        train_features, train_target, scaler = load_and_preprocess_data(train_file)
        
        # Train model
        dataset_name = os.path.splitext(os.path.basename(train_file))[0]
        model = train_model(train_features, train_target, **HYPERPARAMETERS)

        # Plot training losses
        plot_training_losses(model, dataset_name, output_dir)

        # Load test data
        test_features, test_target, _ = load_and_preprocess_data(test_file, scaler=scaler)

        # Evaluate model
        train_pred, train_rmse = evaluate_model(model, train_features, train_target)
        test_pred, test_rmse = evaluate_model(model, test_features, test_target)

        # Plot actual vs predicted values for the test set
        plot_actual_vs_predicted(test_target, test_pred, dataset_name, output_dir)

         # Save predictions to CSV
        save_predictions_to_csv(test_target, test_pred, test_file, dataset_name, output_dir)

        # Save results
        comparison_results["Dataset"].append(dataset_name)
        comparison_results["Train_RMSE"].append(train_rmse)
        comparison_results["Test_RMSE"].append(test_rmse)

    # Save RMSE comparison
    results_df = pd.DataFrame(comparison_results)
    results_filepath = os.path.join(output_dir, "train_test_comparison_results.csv")
    results_df.to_csv(results_filepath, index=False)
    print(f"Comparison results saved to {results_filepath}")

    # Save RMSE comparison plot
    rmse_plot_filepath = os.path.join(output_dir, "rmse_comparison_plot.png")
    plot_rmse_comparison(comparison_results, rmse_plot_filepath)

if __name__ == "__main__":
    main()
