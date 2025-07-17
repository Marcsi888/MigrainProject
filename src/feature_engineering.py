import pandas as pd
from sklearn.preprocessing import StandardScaler
def encode_categorical(df, categorical_cols):
    """One-hot encoding categorical vars"""
    return pd.get_dumies(df, columns=categorical_cols, drop_first=True)

def normalize_features(df, num_cols):
    """Standardizing numerical features"""
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df