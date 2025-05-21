import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    return df

def split_and_scale(df: pd.DataFrame):
    df = encode_features(df)
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    numeric_cols = X.select_dtypes(include='number').columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test

def load_and_preprocess(raw_filename: str='survey_lung_cancer.csv'):
    raw_path = os.path.join(RAW_DATA_DIR, raw_filename)
    df = pd.read_csv(raw_path)
    
    X_train, X_test, y_train, y_test = split_and_scale(df)
    
    df_proc = encode_features(df)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    proc_path = os.path.join(PROCESSED_DATA_DIR, raw_filename)
    df_proc.to_csv(proc_path, index=False)
    print(f"Processed data saved to {proc_path}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print("Train/test split ready:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
