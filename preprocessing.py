import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop unwanted column
    df = df.drop('crash_date', axis=1)

    # Remove data leakage columns
    leak_cols = [
        'injuries_total',
        'injuries_fatal',
        'injuries_incapacitating',
        'injuries_non_incapacitating',
        'injuries_reported_not_evident',
        'injuries_no_indication'
    ]
    df = df.drop(leak_cols, axis=1)

    # Identify columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Encode categorical data
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Feature scaling
    scaler = StandardScaler()
    target = 'most_severe_injury'
    num_cols = [col for col in num_cols if col != target]

    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df