import pandas as pd

def load_data(data_path):
    """Load and preprocess data from CSV file"""
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp')
    return df.reset_index(drop=True)