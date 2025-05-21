import os
import pandas as pd
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_data(filename: str) -> pd.DataFrame:
    """
    Загружает CSV-файл из папки data/raw и возвращает DataFrame.
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Не найден файл {filepath}")
    return pd.read_csv(filepath)

def save_processed_data(df: pd.DataFrame, filename: str):
    """
    Сохраняет DataFrame в CSV в папку data/processed.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Обработанные данные сохранены в {output_path}")

if __name__ == "__main__":
    # Пример использования: подставьте имя вашего CSV-файла в data/raw
    raw_fn = "survey_lung_cancer.csv"
    df = load_data(raw_fn)
    print(df.head())
    # Сохраним исходные данные в папке processed (можно переименовать)
    save_processed_data(df, raw_fn.replace(" ", "_"))
