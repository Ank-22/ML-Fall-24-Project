

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

# Constants - you'll need to update these paths
TRAIN_FILE_PATHS = [
    'Dataset/INFY_DATA.csv', 
    'Dataset/LTIM_DATA.csv', 
    'Dataset/TCS_DATA.csv', 
    'Dataset/WIPRO_DATA.csv', 
    'Dataset/PERSISTENT_DATA.csv'
]
TEST_FILE_PATHS = ['Test_Dataset/INFY_DATA_TEST.csv',
     'Test_Dataset/LTIM_DATA_TEST.csv',
     'Test_Dataset/TCS_DATA_TEST.csv',
    'Test_Dataset/WIPRO_DATA_TEST.csv',
    'Test_Dataset/PERSISTENT_DATA_TEST.csv']

COLUMNS_TO_FILL = ['Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20', 
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']
FEATURE_COLUMNS = ['Open', 'Volume', 'PE_Ratio', '52_Week_High', '52_Week_Low',
                   'Is_52_week_high', 'Is_52_week_low', 'Is_high', 'Is_low',
                   'Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20',
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']

def load_and_preprocess_data(file_path, lag_features=None, num_lags=3, correlation_threshold=0.15, scaler=None):
    # Load data
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['hour'] = data['Datetime'].dt.hour
    data['minute'] = data['Datetime'].dt.minute
    data['dayofweek'] = data['Datetime'].dt.dayofweek

    # Fill missing values
    data[COLUMNS_TO_FILL] = data[COLUMNS_TO_FILL].fillna(method='ffill')

    # Correlation check 
    if lag_features is None:
        corr_matrix = data.corr()
        target_corr = corr_matrix['Target'].abs()
        lag_features = target_corr[target_corr > correlation_threshold].index.tolist()
        lag_features.remove('Target')  

    # Create lagged features
    if lag_features:
        data = create_lagged_features(data, lag_features=lag_features, num_lags=num_lags)

    # Scale features
    feature_columns = FEATURE_COLUMNS + [f"{col}_lag{lag}" for col in lag_features for lag in range(1, num_lags + 1)]
    
    # Use provided scaler or create a new one
    if scaler is None:
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(data[feature_columns])
    else:
        features_scaled = scaler.transform(data[feature_columns])
    
    features = pd.DataFrame(features_scaled, columns=feature_columns)
    target = data['Target']

    return features, target, scaler, feature_columns, lag_features

def create_lagged_features(data, lag_features, num_lags=3):
    for feature in lag_features:
        for lag in range(1, num_lags + 1):
            data[f"{feature}_lag{lag}"] = data[feature].shift(lag)

    # Drop rows with NaN values caused by lagging
    data_with_lags = data.dropna().reset_index(drop=True)
    return data_with_lags

def train_model(X_train, y_train, **kwargs):
    """Train XGBoost model on full dataset"""
    model = XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        **kwargs  # Pass additional hyperparameters
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return RMSE and predictions"""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    return y_pred, rmse

def plot_rmse_comparison(comparison_results, filepath):
    """
    Plot RMSE comparison for training and testing datasets.

    Parameters:
    - comparison_results: Dictionary or DataFrame with RMSE values for training and testing.
    - filepath: Path to save the plot.
    """
    # Convert to DataFrame for easier handling
    results_df = pd.DataFrame(comparison_results)
    
    # Extract dataset names, train RMSE, and test RMSE
    datasets = results_df["Dataset"]
    train_rmse = results_df["Train_RMSE"]
    test_rmse = results_df["Test_RMSE"]
    
    # Define x positions for bars
    x = np.arange(len(datasets))
    width = 0.35  # Width of the bars

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_rmse, width=width, label='Train RMSE', color='skyblue', alpha=0.8)
    plt.bar(x + width/2, test_rmse, width=width, label='Test RMSE', color='orange', alpha=0.8)

    # Add labels, title, and legend
    plt.xlabel("Dataset")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison: Train vs Test")
    plt.xticks(x, datasets, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(filepath)
    plt.close()
    print(f"RMSE comparison plot saved to {filepath}")

def plot_actual_vs_predicted(y_test, y_pred, model_name, filepath):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual (Test Set)", alpha=0.7)
    plt.plot(y_pred, label=f"Predicted ({model_name})", alpha=0.7)
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Index-based plot saved to '{filepath}'.")

def main():
    # Create output directory
    output_dir = "results_train_test_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare results storage
    comparison_results = {
        "Dataset": [],
        "Train_RMSE": [],
        "Test_RMSE": [],
        "Train_Predictions_Filepath": [],
        "Test_Predictions_Filepath": []
    }

    # Process each training dataset
    for train_file, test_file in zip(TRAIN_FILE_PATHS, TEST_FILE_PATHS):
        print(f"\nProcessing {train_file} with test file {test_file}...")
        
        # Prepare training data with lagged features
        train_features, train_target, scaler, feature_columns, lag_features = load_and_preprocess_data(
            train_file, 
            lag_features=FEATURE_COLUMNS, 
            num_lags=3
        )

        # Train the model on full training dataset
        model = train_model(train_features, train_target)

        # Evaluate on training data
        train_pred, train_rmse = evaluate_model(model, train_features, train_target)

        # Prepare test data using the same scaler and feature preprocessing
        test_features, test_target, _, _, _ = load_and_preprocess_data(
            test_file, 
            lag_features=FEATURE_COLUMNS, 
            num_lags=3,
            scaler=scaler  # Use the scaler from training data
        )

        # Evaluate on test data
        test_pred, test_rmse = evaluate_model(model, test_features, test_target)

        # Plot results
        dataset_name = os.path.splitext(os.path.basename(train_file))[0]
        
        # Plot training predictions
        train_plot_filepath = os.path.join(output_dir, f"{dataset_name}_train_actual_vs_predicted.png")
        plot_actual_vs_predicted(train_target, train_pred, f"{dataset_name}_Train", train_plot_filepath)
        
        # Plot test predictions
        test_plot_filepath = os.path.join(output_dir, f"{dataset_name}_test_actual_vs_predicted.png")
        plot_actual_vs_predicted(test_target, test_pred, f"{dataset_name}_Test", test_plot_filepath)

        # Store results
        comparison_results["Dataset"].append(dataset_name)
        comparison_results["Train_RMSE"].append(train_rmse)
        comparison_results["Test_RMSE"].append(test_rmse)
        comparison_results["Train_Predictions_Filepath"].append(train_plot_filepath)
        comparison_results["Test_Predictions_Filepath"].append(test_plot_filepath)

    # Save comparison results
    results_df = pd.DataFrame(comparison_results)
    results_filepath = os.path.join(output_dir, "train_test_comparison_results.csv")
    results_df.to_csv(results_filepath, index=False)
    print(f"\nComparison results saved to {results_filepath}")

    # Save RMSE comparison plot
    rmse_plot_filepath = os.path.join(output_dir, "rmse_comparison_plot.png")
    plot_rmse_comparison(comparison_results, rmse_plot_filepath)
    print(f"\nRMSE comparison plot saved to {rmse_plot_filepath}")

if __name__ == "__main__":
    main()