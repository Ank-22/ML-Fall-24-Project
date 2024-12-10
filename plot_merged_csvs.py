import pandas as pd
import matplotlib.pyplot as plt

# Function to read multiple CSVs and plot Predictions vs Actuals in a single plot
def plot_combined_csvs_with_colors(csv_files, prediction_colors, actual_color, prediction_labels, title):
    """
    Reads multiple CSV files and plots their Predictions and a single Actuals column in one plot.

    Parameters:
    csv_files (list): List of file paths to CSV files.
    prediction_colors (list): List of colors for the Prediction plots (one per CSV file).
    actual_color (str): Color for the Actuals plot.
    """
    if len(prediction_colors) != len(csv_files):
        raise ValueError("Number of prediction colors must match the number of CSV files.")
    
    plt.figure(figsize=(12, 8))
    
    actuals_plotted = False  # Flag to ensure 'Actuals' are plotted only once
    for idx, csv_file in enumerate(csv_files):
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Generate x-axis values as count
        x = range(len(df))  # Create an index for x-axis (0, 1, 2, ..., len(df)-1)
        
        # Plot Predictions
        plt.plot(x, df['predictions'], label=prediction_labels[idx], color=prediction_colors[idx], linestyle='-')
        
        # Plot Actuals only once
        if not actuals_plotted:
            plt.plot(x, df['actuals'], label='Actuals', color=actual_color, linestyle='-', linewidth=1)
            actuals_plotted = True

    # Add labels, title, and legend
    plt.xlabel("Timestamp")
    plt.ylabel("Values")
    plt.title("Predictions vs Actuals (" + title + ")")
    plt.legend()
    
    # Show grid and plot
    plt.grid()
    plt.show()

# Example usage
csv_files = [
    "predictions_vs_actuals_HCL.csv",
    "LSTM/TRAIN_HCL_SGD_predictions.csv",
    "Xgboost/HCL_DATA_predictions.csv"
]
prediction_labels = ['GRU','LSTM','Xgboost']  # Specify labels for predictions
prediction_colors = ['purple', 'orange' , 'black']  # Specify colors for predictions
actual_color = 'red'  # Specify color for actuals
title = 'HCL'  # Specify title for the plot
plot_combined_csvs_with_colors(csv_files, prediction_colors, actual_color, prediction_labels,title)
