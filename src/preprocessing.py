import pandas as pd

def load_migraine_data(filepath: str) -> pd.DataFrame:
    """Load migraine dataset from CSV."""
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop NAs, normalize column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna()
    return df
