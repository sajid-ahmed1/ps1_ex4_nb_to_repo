import pandas as pd

def load_data(path: str):
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        path: The file path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(path)

