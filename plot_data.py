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
    plt.plot(x, df['Prediction'], label='Predictions', linestyle='-')
    plt.plot(x, df['Actual'], label='Actuals', linestyle='-')
    
    # Add labels, title, and legend
    plt.xlabel("Timestamp")
    plt.ylabel("Values")
    plt.title("Predictions vs Actuals (TCS)")
    plt.legend()
    
    # Show the plot
    plt.grid()
    plt.show()

# Example usage
plot_csv_data("TCS_last_pred.csv")

"""
OUTPUT :
INFY: 
================================
train - RMSE loss was around 4.2
Test Loss: 22.1403
Mean Absolute Error (MAE): 19.2509
Mean Absolute Percentage Error (MAPE): 1.00%
R² Score: 0.3004
Directional Accuracy: 47.62%

train : 
Test Loss: 12.9822
Mean Absolute Error (MAE): 9.8443
Mean Absolute Percentage Error (MAPE): 0.53%
R² Score: 0.9299
Directional Accuracy: 48.33%

LTIM :
================================
train - RMSE loss was around 7.9784
Test Loss: 52.9046
Mean Absolute Error (MAE): 44.1983
Mean Absolute Percentage Error (MAPE): 0.71%
R² Score: 0.0659
Directional Accuracy: 47.54%

train : 
Test Loss: 32.4086
Mean Absolute Error (MAE): 18.9411
Mean Absolute Percentage Error (MAPE): 0.31%
R² Score: 0.9537
Directional Accuracy: 48.47%

PERSISTENT :
================================
train - RMSE loss was around 10.8
Test Loss: 46.8741
Mean Absolute Error (MAE): 44.4253
Mean Absolute Percentage Error (MAPE): 0.75%
R² Score: 0.4483



TCS : 
================================
Train - RMSE loss was around 20.3
Test Loss: 91.9158
Mean Absolute Error (MAE): 84.0729
Mean Absolute Percentage Error (MAPE): 1.94%
R² Score: -4.5958
Directional Accuracy: 48.41%

train : 
Test Loss: 46.9362
Mean Absolute Error (MAE): 32.9197
Mean Absolute Percentage Error (MAPE): 0.78%
R² Score: 0.8289
Directional Accuracy: 49.55%

WIPRO :
================================
Train - RMSE loss was around 1.2
Test Loss: 1.9692
Mean Absolute Error (MAE): 1.5456
Mean Absolute Percentage Error (MAPE): 0.26%
R² Score: 0.8677
Directional Accuracy: 44.88%

train : 
Test Loss: 2.4865
Mean Absolute Error (MAE): 2.0027
Mean Absolute Percentage Error (MAPE): 0.35%
R² Score: 0.9654
Directional Accuracy: 45.05%
"""