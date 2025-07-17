from src.preprocessing import load_migraine_data, clean_data
from src.feature_engineering import encode_categorical, normalize_features
from src.model_training import train_random_forest

from sklearn.model_selection import train_test_split

df = load_migraine_data("data/raw/migraine.csv")
df = clean_data(df)

# Feature engineering
categorical_cols = ['gender', 'trigger']  # <- change as needed
numerical_cols = ['age', 'pain_level']
df = encode_categorical(df, categorical_cols)
df = normalize_features(df, numerical_cols)

# Split + train
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_random_forest(X_train, y_train)