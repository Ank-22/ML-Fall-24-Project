import pandas as pd

def merge_csv_files(input_files, output_file):
    """
    Merge multiple CSV files with the same columns into one CSV file.

    Parameters:
    input_files (list): List of file paths for the input CSV files.
    output_file (str): File path for the merged output CSV file.
    """
    # Read all CSV files into DataFrames and concatenate them
    dataframes = [pd.read_csv(file) for file in input_files]
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV file saved to: {output_file}")

# Example usage
input_files = [
    "Dataset/INFY_DATA.csv",
    "Dataset/LTIM_DATA.csv",
    "Dataset/TCS_DATA.csv",
    "Dataset/WIPRO_DATA.csv",
    "Dataset/PERSISTENT_DATA.csv"
]
output_file = "Dataset/5_COMPANIES.csv"
merge_csv_files(input_files, output_file)
