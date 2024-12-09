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
FINAL_TEST_FILE = "Test_Dataset/HCL_DATA_TEST.csv"  # Replace with the actual file path
COLUMNS_TO_FILL = ['Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20',
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']
FEATURE_COLUMNS = ['Open', 'Volume', 'PE_Ratio', '52_Week_High', '52_Week_Low',
                   'Is_52_week_high', 'Is_52_week_low', 'Is_high', 'Is_low',
                   'Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20',
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']


# ---- Preprocessing Functions ----
def load_and_preprocess_data(file_path, lag_features=None, num_lags=3, correlation_threshold=0.15, scaler=None):
    # Load data
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['hour'] = data['Datetime'].dt.hour
    data['minute'] = data['Datetime'].dt.minute
    data['dayofweek'] = data['Datetime'].dt.dayofweek

    # Fill missing values
    data[COLUMNS_TO_FILL] = data[COLUMNS_TO_FILL].fillna(method='ffill')

    # Determine lagged features
    if lag_features is None:
        corr_matrix = data.corr()
        target_corr = corr_matrix['Target'].abs()
        lag_features = target_corr[target_corr > correlation_threshold].index.tolist()
        if 'Target' in lag_features:
            lag_features.remove('Target')

    # Create lagged features
    if lag_features:
        data = create_lagged_features(data, lag_features=lag_features, num_lags=num_lags)

    # Prepare feature columns
    feature_columns = FEATURE_COLUMNS + [f"{col}_lag{lag}" for col in lag_features for lag in range(1, num_lags + 1)]
    feature_columns = [col for col in feature_columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

    # Scale features
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
    return data.dropna().reset_index(drop=True)


# ---- Model Training and Evaluation Functions ----
def train_model(X_train, y_train, xgb_model=None, **kwargs):
    """
    Train or incrementally update an XGBoost model.
    """
    model = XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        **kwargs
    )
    model.fit(X_train, y_train, xgb_model=xgb_model)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return RMSE and predictions.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"RMSE: {rmse:.4f}")
    return y_pred, rmse


def plot_actual_vs_predicted(y_test, y_pred, model_name, filepath):
    """
    Plot actual vs predicted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", alpha=0.7, color="blue")
    plt.plot(y_pred, label="Predicted", alpha=0.7, color="orange")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Actual vs Predicted: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to {filepath}")


# ---- Main Function ----
def incremental_train_model(train_files, test_file, lag_features=None, num_lags=3, correlation_threshold=0.15):
    """
    Incrementally train a model on datasets from multiple companies and evaluate on a final test dataset.
    """
    scaler = None  # Initialize scaler for consistent scaling across datasets
    model = None   # Initialize the model

    for idx, file_path in enumerate(train_files):
        print(f"Processing training on dataset: {file_path}...")
        train_features, train_target, scaler, _, _ = load_and_preprocess_data(
            file_path, lag_features=lag_features, num_lags=num_lags, correlation_threshold=correlation_threshold, scaler=scaler
        )

        if model is None:
            print(f"Initializing and training model on {file_path}...")
            model = train_model(train_features, train_target)
        else:
            print(f"Incrementally training model on {file_path}...")
            model = train_model(train_features, train_target, xgb_model=model.get_booster())

    print(f"Evaluating on final test dataset: {test_file}...")
    test_features, test_target, _, _, _ = load_and_preprocess_data(
        test_file, lag_features=lag_features, num_lags=num_lags, scaler=scaler
    )
    test_pred, test_rmse = evaluate_model(model, test_features, test_target)

    return model, test_rmse, test_pred, test_target


def main():
    output_dir = "results_incremental_training"
    os.makedirs(output_dir, exist_ok=True)

    # Incrementally train on all datasets
    model, test_rmse, test_pred, test_target = incremental_train_model(
        TRAIN_FILE_PATHS, FINAL_TEST_FILE, lag_features=FEATURE_COLUMNS, num_lags=3
    )

    print(f"Final Test RMSE: {test_rmse:.4f}")

    # Plot results for the final test dataset
    plot_filepath = os.path.join(output_dir, "final_test_actual_vs_predicted.png")
    plot_actual_vs_predicted(test_target, test_pred, "Final Test Dataset", plot_filepath)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
