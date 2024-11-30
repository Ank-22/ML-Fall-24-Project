import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define stock ticker, start date, and interval
ticker = "^CNXIT"
start_date = "2024-10-29"  # Start date for data collection
end_date = "2024-11-26"    # End date for data collection
interval = "1m"            # Interval (e.g., "1m", "5m")

# Function to generate 8-day chunks within the date range
def generate_date_chunks(start, end, chunk_size=8):
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_size), end)
        yield current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")
        current_start = current_end

# Convert start_date and end_date to datetime objects
start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

# Fetch data in chunks and combine
data_frames = []
for chunk_start, chunk_end in generate_date_chunks(start_date_dt, end_date_dt):
    print(f"Fetching data for {ticker} from {chunk_start} to {chunk_end}...")
    data = yf.download(ticker, start=chunk_start, end=chunk_end, interval=interval)
    if not data.empty:
        data_frames.append(data)
    else:
        print(f"No data retrieved for {chunk_start} to {chunk_end}.")

# Combine all data frames
if data_frames:
    combined_data = pd.concat(data_frames)
    # Save combined data to a CSV file
    output_filename = f"{ticker}_30_days_data_1m.csv"
    combined_data.to_csv(output_filename)
    print(f"Data saved to {output_filename}")
else:
    print("No data retrieved for the specified date range.")
