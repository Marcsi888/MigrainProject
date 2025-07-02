import pandas as pd

def load_migraine_data(file_path):
    """Load the migraine dataset from a CSV file."""
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df

def check_missing_values(df):
    """Print missing values in the DataFrame."""
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found in the dataset")

def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame and return the cleaned DataFrame."""
    duplicates = df.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")
    if duplicates.sum() > 0:
        print("Duplicate rows:")
        print(df[duplicates])
    df_clean = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after removing duplicates: {df_clean.shape}")
    return df_clean

def detect_outliers(df, column):
    """Detect outliers in a numeric column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers in {column}:")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Bounds: [{lower_bound}, {upper_bound}]")
    if len(outliers) > 0:
        print(outliers[[column, 'Type']].head(5))
    print("\n")
    return outliers, lower_bound, upper_bound 