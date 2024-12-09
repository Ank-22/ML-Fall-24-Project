import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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

# ---- Data Preprocessing Functions ----
def load_and_preprocess_data(file_path):
    """Load data, fill missing values, and scale features."""
    # Load data
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data['hour'] = data['Datetime'].dt.hour
    data['minute'] = data['Datetime'].dt.minute
    data['dayofweek'] = data['Datetime'].dt.dayofweek

    # Fill missing values
    data[COLUMNS_TO_FILL] = data[COLUMNS_TO_FILL].fillna(method='ffill')
    
    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(data[FEATURE_COLUMNS])
    features = pd.DataFrame(features_scaled, columns=FEATURE_COLUMNS)
    target = data['Target']
    
    return features, target, data

def split_data(features, target, test_size=0.2):
    """Split data into train, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    return X_train, X_val, X_test, y_train, y_val, y_test

# ---- Model Training and Evaluation Functions ----
def train_model(X_train, y_train, X_val, y_val, early_stopping_rounds=None, **kwargs):
    """
    Train XGBoost model with or without early stopping.
    Accepts additional hyperparameters via **kwargs.
    
    Parameters:
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - early_stopping_rounds: Number of early stopping rounds.
    - **kwargs: Additional hyperparameters for XGBoost.
    
    Returns:
    - model: Trained XGBoost model.
    """
    model = XGBRegressor(
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
        **kwargs  # Pass additional hyperparameters
    )
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X=X_train,
        y=y_train,
        eval_set=eval_set,
        verbose=True
    )
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return RMSE and predictions."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    return y_pred, rmse

def run_grid_search(X_train, y_train):
    """Run grid search to find the best hyperparameters for the model."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    model = XGBRegressor(random_state=42, eval_metric='rmse')
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


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

# ---- Save Plot Functions ----
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

# ---- Save Results Functions ----
def save_to_csv(data, filename):

    if isinstance(data, pd.DataFrame):
        data.to_csv(filename, index=False)
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
    else:
        raise ValueError("Input data must be a pandas DataFrame or a dictionary.")
    
    print(f"Data saved to '{filename}'.")

def plot_feature_importance(model, feature_names, model_name, filepath):

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
    plt.title(f"Feature Importance: {model_name}")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Feature importance plot saved to '{filepath}'.")

def plot_rmse_comparison(results_df, filepath):

    x = np.arange(len(results_df['Company']))  # X positions for the companies
    width = 0.25  # Width of each bar

    plt.figure(figsize=(12, 6))

    # Plot RMSE for Default Parameters
    plt.bar(
        x - width,
        results_df['RMSE (Default)'],
        width=width,
        label='Default Parameters',
        color='skyblue',
        alpha=0.8,
    )

    # Plot RMSE for Tuned Parameters
    plt.bar(
        x,
        results_df['RMSE (Tuned)'],
        width=width,
        label='Tuned Parameters',
        color='orange',
        alpha=0.8,
    )

    # Plot RMSE for Averaged Parameters
    plt.bar(
        x + width,
        results_df['RMSE (Averaged)'],
        width=width,
        label='Averaged Parameters',
        color='green',
        alpha=0.8,
    )

    # Add labels, title, and legend
    plt.xlabel("Company")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison: Default vs Tuned vs Averaged Parameters")
    plt.xticks(x, results_df['Company'], rotation=45, ha="right")  # Rotate company names for readability
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(filepath)
    plt.close()
    print(f"RMSE comparison plot saved to '{filepath}'.")

def main():
    all_best_params = []  # To store the best parameters for each company
    results = []  # To store RMSE results for comparison
    hyperparameter_results = []  # To store hyperparameter results
    output_dir = "results_hyperparameter_tunning"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Process each company
    for file_path in FILE_PATH:
        print(f"Processing {file_path}...")
        features, target, data = load_and_preprocess_data(file_path)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target)

        # Train using default parameters
        print(f"Training with default parameters for {file_path}...")
        default_model = train_model(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
        y_pred_default, rmse_default = evaluate_model(default_model, X_test, y_test)

        # Save plots for default parameters
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        plot_actual_vs_predicted_individual(
            y_test,
            y_pred_default,
            f"{base_filename} (Default Parameters)",
            os.path.join(output_dir, f"{base_filename}_default_actual_vs_predicted.png"),
        )

        # Perform hyperparameter tuning
        print(f"Performing grid search for {file_path}...")
        best_params = run_grid_search(X_train, y_train)
        print(f"Best Parameters for {file_path}: {best_params}")
        all_best_params.append(best_params)

        # Train using tuned parameters
        print(f"Training with tuned parameters for {file_path}...")
        tuned_model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            early_stopping_rounds=10,
            **best_params
        )
        y_pred_tuned, rmse_tuned = evaluate_model(tuned_model, X_test, y_test)

        # Save plots for tuned parameters
        plot_actual_vs_predicted_individual(
            y_test,
            y_pred_tuned,
            f"{base_filename} (Tuned Parameters)",
            os.path.join(output_dir, f"{base_filename}_tuned_actual_vs_predicted.png"),
        )

        # Collect results for comparison
        results.append({
            'Company': base_filename,
            'RMSE (Default)': rmse_default,
            'RMSE (Tuned)': rmse_tuned,
            'RMSE (Averaged)': None  # Placeholder for now
        })

        # Add tuned parameters to hyperparameter results
        hyperparameter_results.append({
            "Company": base_filename,
            "Tuned Parameters": best_params,
            "Averaged Parameters": None  # Placeholder for now
        })

    # Step 2: Calculate averaged parameters after all companies are processed
    print("Calculating averaged parameters...")
    averaged_params = {
        key: np.mean([params[key] for params in all_best_params])
        for key in all_best_params[0]
    }
    averaged_params['n_estimators'] = int(averaged_params['n_estimators'])
    averaged_params['max_depth'] = int(averaged_params['max_depth'])
    print(f"Averaged Hyperparameters: {averaged_params}")

    # Step 3: Train and evaluate using averaged parameters
    for idx, file_path in enumerate(FILE_PATH):
        print(f"Training model for {file_path} using averaged parameters...")
        features, target, data = load_and_preprocess_data(file_path)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target)

        model_averaged = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            early_stopping_rounds=10,
            **averaged_params
        )
        y_pred_averaged, rmse_averaged = evaluate_model(model_averaged, X_test, y_test)

        # Save plots for averaged parameters
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        plot_actual_vs_predicted_individual(
            y_test,
            y_pred_averaged,
            f"{base_filename} (Averaged Parameters)",
            os.path.join(output_dir, f"{base_filename}_averaged_actual_vs_predicted.png"),
        )

        # Update RMSE (Averaged) in results
        results[idx]['RMSE (Averaged)'] = rmse_averaged

        # Add averaged parameters to hyperparameter results
        if idx == len(FILE_PATH) - 1:  # Only add averaged parameters once
            hyperparameter_results[idx]['Averaged Parameters'] = averaged_params

    # Step 4: Save RMSE comparison results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "rmse_comparison.csv"), index=False)
    print("RMSE comparison results saved to 'rmse_comparison.csv'.")

    # Save hyperparameter results
    print("Saving hyperparameter results...")
    tuned_averaged_params_df = pd.DataFrame([
        {
            "Company": entry["Company"],
            "Tuned Parameters": entry["Tuned Parameters"],
            "Averaged Parameters": entry["Averaged Parameters"] if entry["Averaged Parameters"] else "N/A"
        }
        for entry in hyperparameter_results
    ])
    tuned_averaged_params_df.to_csv(os.path.join(output_dir, "hyperparameter_results.csv"), index=False)
    print("Hyperparameter results saved to 'hyperparameter_results.csv'.")

    # Optional: Plot RMSE comparison
    plot_rmse_comparison(results_df, os.path.join(output_dir, "rmse_comparison.png"))


if __name__ == "__main__":
    main()
