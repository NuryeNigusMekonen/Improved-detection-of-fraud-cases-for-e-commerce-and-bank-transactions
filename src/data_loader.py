import pandas as pd
import os

def load_csv(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise pd.errors.EmptyDataError(f"File is empty: {file_path}")
        return df
    except pd.errors.ParserError as e:
        raise ValueError(f"Parsing error in file: {file_path}\n{e}")
    except Exception as e:
        raise Exception(f"Error loading file: {file_path}\n{e}")
