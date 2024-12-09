import pandas as pd
import matplotlib.pyplot as plt

# Function to read a CSV and plot its data
def plot_csv_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Generate x-axis values as count
    x = range(len(df))  # Create an index for x-axis (0, 1, 2, ..., len(df)-1)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x, df['Prediction'], label='Predictions', marker='o')
    plt.plot(x, df['Actual'], label='Actuals', marker='x')
    
    # Add labels, title, and legend
    plt.xlabel("Per Minute")
    plt.ylabel("Values")
    plt.title("Predictions vs Actuals")
    plt.legend()
    
    # Show the plot
    plt.grid()
    plt.show()

# Example usage
plot_csv_data("wandb_export_2024-12-08T18_28_16.854-05_00.csv")

"""
OUTPUT :
INFY: 
train - RMSE loss was around 4.2
Test Loss: 22.1403
Mean Absolute Error (MAE): 19.2509
Mean Absolute Percentage Error (MAPE): 1.00%
R² Score: 0.3004
Directional Accuracy: 47.62%

LTIM :
train - RMSE loss was around 7.9784
Test Loss: 52.9046
Mean Absolute Error (MAE): 44.1983
Mean Absolute Percentage Error (MAPE): 0.71%
R² Score: 0.0659
Directional Accuracy: 47.54%

PERSISTENT :
train - RMSE loss was around 10.8
Test Loss: 46.8741
Mean Absolute Error (MAE): 44.4253
Mean Absolute Percentage Error (MAPE): 0.75%
R² Score: 0.4483

TCS : 
Train - RMSE loss was around 67.8
Test Loss: 184.6662
Mean Absolute Error (MAE): 180.5320
Mean Absolute Percentage Error (MAPE): 4.18%
R² Score: -21.5870

WIPRO :
Train - RMSE loss was around 1.2
Test Loss: 1.9692
Mean Absolute Error (MAE): 1.5456
Mean Absolute Percentage Error (MAPE): 0.26%
R² Score: 0.8677
Directional Accuracy: 44.88%
"""