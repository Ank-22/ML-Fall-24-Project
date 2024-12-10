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


def train_model(X_train, y_train, X_val, y_val, early_stopping_rounds=None, **kwargs):
    """Train XGBoost model with or without early stopping."""

    model = XGBRegressor(
        random_state=42,
        eval_metric="rmse",
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

def save_combined_training_losses(model_with_early_stopping, model_without_early_stopping, title, filepath):
    """Save combined training losses plot for early stopping and no early stopping."""
    results_with_early_stopping = model_with_early_stopping.evals_result()
    results_without_early_stopping = model_without_early_stopping.evals_result()

    # Get training losses only
    train_losses_with_early_stopping = results_with_early_stopping['validation_0']['rmse']
    train_losses_without_early_stopping = results_without_early_stopping['validation_0']['rmse']

    epochs_with_early_stopping = len(train_losses_with_early_stopping)
    epochs_without_early_stopping = len(train_losses_without_early_stopping)

    x_axis_with_early_stopping = range(0, epochs_with_early_stopping)
    x_axis_without_early_stopping = range(0, epochs_without_early_stopping)

    plt.figure(figsize=(10, 6))

    # Plot training losses
    plt.plot(
        x_axis_with_early_stopping,
        train_losses_with_early_stopping,
        label="Train Loss (Early Stopping)"
    )
    plt.plot(
        x_axis_without_early_stopping,
        train_losses_without_early_stopping,
        label="Train Loss (No Early Stopping)"
    )

    # Add labels, title, and legend
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Combined training losses plot saved to '{filepath}'.")


def plot_actual_vs_predicted_combined(y_test, y_pred_early, y_pred_no_early, model_name, filepath):
    """Plot Actual vs Predicted for Early Stopping and No Early Stopping."""
    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.plot(y_test.values, label="Actual (Test Set)", alpha=0.7)

    # Plot predictions with early stopping
    plt.plot(y_pred_early, label="Predicted (Early Stopping)", alpha=0.7)

    # Plot predictions without early stopping
    plt.plot(y_pred_no_early, label="Predicted (No Early Stopping)", alpha=0.7)

    # Add labels, title, and legend
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Actual vs Predicted: Early Stopping vs No Early Stopping : {model_name}")
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath)
    plt


def plot_actual_vs_predicted_individual(y_test, y_pred, model_name, filepath):

    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.plot(y_test.values, label="Actual",  alpha=0.7)

    # Plot predictions
    plt.plot(y_pred, label=f"Predicted",  alpha=0.7)

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
    """Save training losses plot as a file."""
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    # Plot only training losses
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"Training losses plot saved to '{filepath}'.")


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

def plot_rmse_comparison(rmse_df, filepath):

    x = np.arange(len(rmse_df['Company'])) 
    width = 0.35  

    plt.figure(figsize=(12, 6))

    # Plot RMSE for Early Stopping
    plt.bar(
        x - width / 2,
        rmse_df['RMSE (Early Stopping)'],
        width=width,
        label='Early Stopping',
        color='skyblue',
        alpha=0.8
    )

    # Plot RMSE for No Early Stopping
    plt.bar(
        x + width / 2,
        rmse_df['RMSE (No Early Stopping)'],
        width=width,
        label='No Early Stopping',
        color='orange',
        alpha=0.8
    )

    # Add labels, title, and legend
    plt.xlabel("Company")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison: Early Stopping vs No Early Stopping")
    plt.xticks(x, rmse_df['Company'], rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig(filepath)
    plt.close()

def main():
    output_dir = "results_early_stopping_comparison"
    os.makedirs(output_dir, exist_ok=True)

    results = []  # Store RMSE results for each company

    for file_path in FILE_PATH:
        print(f"Processing {file_path}...")
        features, target, data = load_and_preprocess_data(file_path)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target)

        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # Retrieve the Datetime values for the test set
        datetime_test = data.loc[X_test.index, 'Datetime']

        # Train with Early Stopping
        print(f"Training {base_filename} with Early Stopping...")
        model_with_early_stopping = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            early_stopping_rounds=50,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            colsample_bytree=0.88,
            subsample=0.84
        )
        save_plot_training_losses(
            model_with_early_stopping,
            f"Losses with Early Stopping ({base_filename})",
            os.path.join(output_dir, f"{base_filename}_early_stopping_losses.png")
        )
        y_pred_early_stopping, rmse_early_stopping = evaluate_model(model_with_early_stopping, X_test, y_test)

        # Save predictions and validation losses for Early Stopping
        early_stopping_results = pd.DataFrame({
            "Datetime": datetime_test.values,
            "Predicted (Early Stopping)": y_pred_early_stopping
        })
        save_to_csv(early_stopping_results, os.path.join(output_dir, f"{base_filename}_early_stopping_predictions.csv"))

        # Train without Early Stopping
        print(f"Training {base_filename} without Early Stopping...")
        model_without_early_stopping = train_model( X_train,
            y_train,
            X_val,
            y_val,
            early_stopping_rounds=None,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            colsample_bytree=0.88,
            subsample=0.84
            )
        save_plot_training_losses(
            model_without_early_stopping,
            f"Losses without Early Stopping ({base_filename})",
            os.path.join(output_dir, f"{base_filename}_no_early_stopping_losses.png")
        )
        y_pred_no_early_stopping, rmse_no_early_stopping = evaluate_model(model_without_early_stopping, X_test, y_test)

        # Save predictions and validation losses for No Early Stopping
        no_early_stopping_results = pd.DataFrame({
            "Datetime": datetime_test.values,
            "Predicted (No Early Stopping)": y_pred_no_early_stopping
        })
        save_to_csv(no_early_stopping_results, os.path.join(output_dir, f"{base_filename}_no_early_stopping_predictions.csv"))

        # Collect RMSE results
        results.append({
            "Company": base_filename,
            "RMSE (Early Stopping)": rmse_early_stopping,
            "RMSE (No Early Stopping)": rmse_no_early_stopping
        })

        # Save actual vs predicted plots
        plot_actual_vs_predicted_individual(
            y_test,
            y_pred_early_stopping,
            f"{base_filename} (Early Stopping)",
            os.path.join(output_dir, f"{base_filename}_actual_vs_predicted_early_stopping.png")
        )
        plot_actual_vs_predicted_individual(
            y_test,
            y_pred_no_early_stopping,
            f"{base_filename} (No Early Stopping)",
            os.path.join(output_dir, f"{base_filename}_actual_vs_predicted_no_early_stopping.png")
        )

        save_combined_training_losses(
            model_with_early_stopping,
            model_without_early_stopping,
            f"Combined Training Losses ({base_filename})",
            os.path.join(output_dir, f"{base_filename}_combined_training_losses.png")
        )


        # Save combined actual vs predicted plot
        plot_actual_vs_predicted_combined(
            y_test,
            y_pred_early_stopping,
            y_pred_no_early_stopping, base_filename,
            os.path.join(output_dir, f"{base_filename}_actual_vs_predicted_combined.png")
        )

        # Save feature importance plots
        plot_feature_importance(
            model_with_early_stopping,
            FEATURE_COLUMNS,
            f"{base_filename} (Early Stopping)",
            os.path.join(output_dir, f"{base_filename}_feature_importance_early_stopping.png")
        )
        plot_feature_importance(
            model_without_early_stopping,
            FEATURE_COLUMNS,
            f"{base_filename} (No Early Stopping)",
            os.path.join(output_dir, f"{base_filename}_feature_importance_no_early_stopping.png")
        )

    # Save RMSE comparison for all companies
    rmse_df = pd.DataFrame(results)
    save_to_csv(rmse_df, os.path.join(output_dir, "rmse_comparison.csv"))
    print("RMSE comparison results saved.")

    # Plot RMSE comparison
    plot_rmse_comparison(rmse_df, os.path.join(output_dir, "rmse_comparison.png"))
    print("RMSE comparison plot saved.")



if __name__ == "__main__":
    main()
