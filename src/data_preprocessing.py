import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path, sep=';')

def preprocess_data(df):
    """Performs preprocessing on the dataframe."""
    # Encode categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Scaling numerical features
    numerical_features = df.select_dtypes(include=np.number).columns.drop('y')
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df