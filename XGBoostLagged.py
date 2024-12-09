import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os
import numpy as np

# Constants
FILE_PATH = ['Dataset/INFY_DATA.csv', 'Dataset/LTIM_DATA.csv', 'Dataset/TCS_DATA.csv', 'Dataset/WIPRO_DATA.csv', 'Dataset/PERSISTENT_DATA.csv']
COLUMNS_TO_FILL = ['Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20', 
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']
FEATURE_COLUMNS = ['Open', 'Volume', 'PE_Ratio', '52_Week_High', '52_Week_Low',
                   'Is_52_week_high', 'Is_52_week_low', 'Is_high', 'Is_low',
                   'Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20',
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']

AVG_HYPERPARAMETERS = {'colsample_bytree': 0.8, 'learning_rate': 0.06, 'max_depth': 5, 'n_estimators': 240, 'subsample': 0.8}

def load_and_preprocess_data(file_path, lag_features=None, num_lags=3, correlation_threshold=0.50):

    # Load data
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['hour'] = data['Datetime'].dt.hour
    data['minute'] = data['Datetime'].dt.minute
    data['dayofweek'] = data['Datetime'].dt.dayofweek

    # Fill missing values
    data[COLUMNS_TO_FILL] = data[COLUMNS_TO_FILL].fillna(method='ffill')

    # Correlation check (optional if lag_features is None)
    if lag_features is None:
        corr_matrix = data.corr()
        target_corr = corr_matrix['Target'].abs()
        lag_features = target_corr[target_corr > correlation_threshold].index.tolist()
        lag_features.remove('Target')  

    # Create lagged features
    if lag_features:
        data = create_lagged_features(data, lag_features=lag_features, num_lags=num_lags)

    # Scale features
    scaler = MinMaxScaler()
    feature_columns = FEATURE_COLUMNS + [f"{col}_lag{lag}" for col in lag_features for lag in range(1, num_lags + 1)]
    features_scaled = scaler.fit_transform(data[feature_columns])
    features = pd.DataFrame(features_scaled, columns=feature_columns)
    target = data['Target']

    return features, target, data

def split_data(features, target, test_size=0.2):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def create_lagged_features(data, lag_features, num_lags=3):
 
    for feature in lag_features:
        for lag in range(1, num_lags + 1):
            data[f"{feature}_lag{lag}"] = data[feature].shift(lag)

    # Drop rows with NaN values caused by lagging
    data_with_lags = data.dropna().reset_index(drop=True)
    return data_with_lags

def get_highly_important_features(model, feature_names, threshold=0.5):

    importance_scores = model.feature_importances_
    important_features = [feature_names[i] for i, score in enumerate(importance_scores) if score > threshold]
    return important_features


def train_model(X_train, y_train, **kwargs):
    """Train XGBoost model without validation set."""
    
    model = XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        **kwargs  # Pass additional hyperparameters
    )
    model.fit(
        X=X_train,
        y=y_train,
        verbose=True
    )
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return RMSE and predictions."""

    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    return y_pred, rmse


def plot_actual_vs_predicted_individual(y_test, y_pred, model_name, filepath):

    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.plot(y_test.values, label="Actual (Test Set)",  alpha=0.7)

    # Plot predictions
    plt.plot(y_pred, label=f"Predicted ({model_name})",  alpha=0.7)

    # Add labels, title, and legend
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Index-based plot saved to '{filepath}'.")



def save_plot_training_losses(model, title, filepath):
    """Save training and validation losses plot as a file."""

    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
    plt.plot(x_axis, results['validation_1']['rmse'], label='Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to '{filepath}'.")


def save_to_csv(data, filename):

    if isinstance(data, pd.DataFrame):
        data.to_csv(filename, index=False)
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    else:
        raise ValueError("Input data must be a pandas DataFrame or a dictionary.")
    
    print(f"Data saved to '{filename}'.")

def plot_feature_importance(model, feature_names, dataset_name, filepath):

    # Get feature importance scores
    importance_scores = model.feature_importances_

    # Sort features by importance
    sorted_idx = importance_scores.argsort()
    sorted_feature_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importance_scores[sorted_idx]

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(sorted_feature_names, sorted_importances, color="skyblue")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title(f"Feature Importance: {dataset_name}")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Feature importance plot saved to '{filepath}'.")

def plot_rmse_comparison(rmse_results, filepath):
    x = np.arange(len(rmse_results['Dataset']))  
    width = 0.35 

    plt.figure(figsize=(12, 6))

    # Plot RMSE for Non-Lagged Features
    plt.bar(
        x - width / 2,
        rmse_results['RMSE (Non-Lagged)'],
        width=width,
        label="Without Lagged Features",
        color="skyblue",
        alpha=0.8,
    )

    # Plot RMSE for Lagged Features
    plt.bar(
        x + width / 2,
        rmse_results['RMSE (Lagged)'],
        width=width,
        label="With Lagged Features",
        color="orange",
        alpha=0.8,
    )

    # Add labels and title
    plt.xlabel("Dataset")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison: Lagged vs Non-Lagged Features")
    plt.xticks(x, rmse_results['Dataset'], rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(filepath)
    plt.close()
    print(f"RMSE comparison plot saved to '{filepath}'.")


def main():
    output_dir = "results_lagged_features"
    os.makedirs(output_dir, exist_ok=True)

    rmse_results = {"Dataset": [], "RMSE (Non-Lagged)": [], "RMSE (Lagged)": []}

    for idx, file_path in enumerate(FILE_PATH):
        print(f"Processing {file_path}...")

        # Without lagged features
        print(f"Training without lagged features for {file_path}...")
        features, target, data = load_and_preprocess_data(file_path, lag_features=[], num_lags=0)  # No lagging
        X_train, X_test, y_train, y_test = split_data(features, target)

        # Train baseline model
        model_nonlagged = train_model(X_train, y_train, **AVG_HYPERPARAMETERS)
        _, rmse_nonlagged = evaluate_model(model_nonlagged, X_test, y_test)

        # Select important features using threshold
        print(f"Selecting important features for lagging based on importance threshold...")
        important_features = get_highly_important_features(model_nonlagged, FEATURE_COLUMNS, threshold=0.25)

        # With lagged features
        print(f"Training with lagged features for {file_path}...")
        features, target, data = load_and_preprocess_data(file_path, lag_features=important_features, num_lags=5)  # Lagging only important features
        X_train, X_test, y_train, y_test = split_data(features, target)

        model_lagged = train_model(X_train, y_train, **AVG_HYPERPARAMETERS)
        _, rmse_lagged = evaluate_model(model_lagged, X_test, y_test)

        # Store results
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        rmse_results["Dataset"].append(dataset_name)
        rmse_results["RMSE (Non-Lagged)"].append(rmse_nonlagged)
        rmse_results["RMSE (Lagged)"].append(rmse_lagged)

        # Plot feature importance for non-lagged model
        importance_plot_path = os.path.join(output_dir, f"{dataset_name}_feature_importance.png")
        plot_feature_importance(
            model_nonlagged, 
            features.columns, 
            dataset_name, 
            filepath=importance_plot_path
        )

    # Save RMSE comparison for all datasets
    rmse_df = pd.DataFrame(rmse_results)
    save_to_csv(rmse_df, os.path.join(output_dir, "rmse_comparison_lagged_vs_nonlagged.csv"))
    print("RMSE comparison results saved.")

    # Plot RMSE comparison for lagged vs non-lagged features
    plot_rmse_comparison(
        rmse_results=rmse_df,
        filepath=os.path.join(output_dir, "rmse_comparison_lagged_vs_nonlagged.png"),
    )
if __name__ == "__main__":
    main()