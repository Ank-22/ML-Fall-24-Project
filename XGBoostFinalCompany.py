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
FINAL_TEST_FILE = "Test_Dataset/HCL_DATA_TEST.csv"  
COLUMNS_TO_FILL = ['Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20',
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']
FEATURE_COLUMNS = ['Open', 'Volume', 'PE_Ratio', '52_Week_High', '52_Week_Low',
                   'Is_52_week_high', 'Is_52_week_low', 'Is_high', 'Is_low',
                   'Maket_index', 'Sector_index', 'SMA_20', 'SMA_50', 'EMA_20',
                   'EMA_50', 'BB_upper', 'BB_lower', 'RSI', 'MACD']

# Hyperparameters
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
    """
    Train an XGBoost model with given hyperparameters.
    """
    model = XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        **kwargs
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return RMSE and predictions.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"RMSE: {rmse:.4f}")
    return y_pred, rmse

def incremental_train_model(train_files, test_file):
    """
    Train a model on multiple training datasets and evaluate on a final test dataset.
    """
    scaler = None  
    model = None   

    for idx, file_path in enumerate(train_files):
        print(f"Processing training on dataset: {file_path}...")
        train_features, train_target, scaler = load_and_preprocess_data(file_path, scaler=scaler)

        if model is None:
            print(f"Initializing and training model on {file_path}...")
            model = train_model(train_features, train_target, **HYPERPARAMETERS)
        else:
            print(f"Incrementally training model on {file_path}...")
            model.fit(train_features, train_target, xgb_model=model.get_booster())

    print(f"Evaluating on final test dataset: {test_file}...")
    test_features, test_target, _ = load_and_preprocess_data(test_file, scaler=scaler)
    test_pred, test_rmse = evaluate_model(model, test_features, test_target)

    return model, test_rmse, test_pred, test_target

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

def save_predictions_to_csv(datetime_values, predicted, output_path):

    results = pd.DataFrame({
        "Datetime": datetime_values,
        "Predicted": predicted
    })
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")



def main():
    output_dir = "results_incremental_training"
    os.makedirs(output_dir, exist_ok=True)

    # Train model on all datasets incrementally and evaluate on final test dataset
    model, test_rmse, test_pred, test_target = incremental_train_model(TRAIN_FILE_PATHS, FINAL_TEST_FILE)

    print(f"Final Test RMSE: {test_rmse:.4f}")

    # Load test dataset to retrieve Datetime column
    test_data = pd.read_csv(FINAL_TEST_FILE)
    test_datetime = pd.to_datetime(test_data["Datetime"])

    # Save predictions to CSV
    predictions_filepath = os.path.join(output_dir, "HCL_test_predictions.csv")
    save_predictions_to_csv(test_datetime, test_pred, predictions_filepath)

    # Plot results for the final test dataset
    plot_filepath = os.path.join(output_dir, "final_test_actual_vs_predicted.png")
    plot_actual_vs_predicted(test_target, test_pred, "Final Test Dataset", plot_filepath)

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
