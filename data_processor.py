import os
import dask.dataframe as dd
import gc

import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import time
from sklearn.linear_model import LinearRegression
import pickle
from datetime import datetime
import zipfile
from typing import Final

DATA_DIR: Final[str] = "~/BIG_DATA"

def extract_zip(archive_path, output_dir):
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def dataframe_loader(path: str) -> dd.DataFrame:
    df = dd.read_csv(os.path.join(DATA_DIR, path), engine="c", on_bad_lines='skip', )
    df = df.astype('float32') # изначальный тип данных датасета был float64, уменьшаем т.к значения не выходят за пределы размерности этого типа
                            # c float16 уже появляются ошибки о переполнении
    return df

async def load_datasets(name: str): # загружаем данные из каждого файла поочередно и выполняем манипуляции с ними
    save_plots_dir = 'plots'  # Директория для сохранения графиков
    os.makedirs(save_plots_dir, exist_ok=True)  # Создаем директорию, если она не существует
    for i in range(4):
        df = dataframe_loader(f"{name}_{i}.csv")
        print(f"{name}_{i} loaded, making describe...")
        yield df
        gc.collect()

def create_new_df(name: str):
    df_list = []  # Список для хранения частей DataFrame
    for df_part in load_datasets(name):
        df_list.append(df_part.compute())
        print("Добавлен df в list")

    # Объединяем все части в один DataFrame
    df = pd.concat(df_list, axis=0, join='inner')
    print("добавлен df в df")
    df.to_csv("~/data/data_X.csv") 
    return df

def drop_columns_with_lowest_correlation(df, n=10, columns_to_drop=None):
    """
    Удаляет столбцы с наименьшей корреляцией из DataFrame и возвращает его.

    Параметры:
    df (pd.DataFrame): Исходный DataFrame.
    n (int): Количество столбцов с наименьшей корреляцией, которые нужно удалить.

    Возвращает:
    pd.DataFrame: DataFrame с удаленными столбцами.
    """

    # Вычисляем матрицу корреляции
    corr_matrix = df.corr()
    
    # Преобразуем матрицу корреляции в одномерный Series, исключая диагональные элементы
    corr_series = corr_matrix.unstack().sort_values().drop_duplicates()
    
    # Исключаем корреляции с самими собой (значения 1.0)
    corr_series = corr_series[corr_series != 1.0]

    # Получаем n пар столбцов с наименьшей корреляцией
    low_corr_pairs = corr_series.head(n).index.tolist()

    if columns_to_drop is None:
        # Собираем уникальные столбцы для удаления
        columns_to_drop = set()
        for pair in low_corr_pairs:
            columns_to_drop.add(pair[0])
            columns_to_drop.add(pair[1])

    # Удаляем столбцы из DataFrame
    df_dropped = df.drop(columns=columns_to_drop)

    return df_dropped, columns_to_drop

def load_target():
    df_list = []
    for _ in range(4):
        df = dd.read_csv(os.path.join(DATA_DIR, f"target_{_}.csv")).astype("float32")
        df = df.compute()
        df = df.dropna() # удаляет nan значения построчно
        df_list.append(df)

    target_df_full = pd.concat(df_list, axis=0)
    print(target_df_full.head())
    target_df_full.to_csv("~/data/data_y.csv")

    return target_df_full, "~/data/data_y.csv"

def load_data_val(columns_to_drop):
    df_val = pd.read_csv(os.path.join(DATA_DIR, "df_val.csv")).astype("float32")
    df_val = df_val.drop(columns=columns_to_drop)
    df_val.to_csv("~/data/data_X_val.csv")
    return df_val, "~/data/data_X_val.csv"

def load_target_val():
    target_val = pd.read_csv(os.path.join(DATA_DIR, "target_val.csv")).astype("float32")
    target_val = target_val.drop([target_val.columns[1]], axis=1)
    target_val.to_csv("~/data/data_y_val.csv")
    return target_val, "~/data/data_y_val.csv"

def linear_regression_test(X_train, y_train, X_test, y_test):
    """
    Обучает модель Linear Regression для задачи регрессии и возвращает модель, предсказания, метрики и время выполнения.

    Параметры:
    - X_train: Признаки обучающей выборки.
    - y_train: Целевая переменная обучающей выборки.
    - X_test: Признаки тестовой выборки.
    - y_test: Целевая переменная тестовой выборки.

    Возвращает:
    - model: Обученная модель LinearRegression.
    - y_pred: Предсказания на тестовой выборке.
    - metrics: Словарь с метриками (R², RMSE, MSE, MAE).
    - execution_time: Словарь с временем выполнения (обучение, предсказание).
    """
    # Создание и обучение модели
    model = LinearRegression()

    # Измерение времени обучения модели
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train

    # Измерение времени предсказания
    start_predict = time.time()
    y_pred = model.predict(X_test)
    end_predict = time.time()
    predict_time = end_predict - start_predict

    # Вычисление метрик
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # RMSE
    r2 = r2_score(y_test, y_pred)  # R²
    mae = mean_absolute_error(y_test, y_pred)  # MAE

    # Вывод метрик
    print(f"Среднеквадратичная ошибка (MSE): {mse}")
    print(f"Среднеквадратичная ошибка (RMSE): {rmse}")
    print(f"Коэффициент детерминации (R²): {r2}")
    print(f"Средняя абсолютная ошибка (MAE): {mae}")

    # Вывод времени выполнения
    print(f"Время обучения модели: {train_time:.2f} секунд")
    print(f"Время предсказания: {predict_time:.2f} секунд")

    # Возвращаем модель, предсказания, метрики и время выполнения
    metrics = {'R²': r2, 'RMSE': rmse, 'MSE': mse, 'MAE': mae}
    execution_time = {'train_time': train_time, 'predict_time': predict_time}
    return model, y_pred, metrics, execution_time

def pipeline(data_X_path: str, data_y_path: str, data_x_valid_path: str, data_y_valid_path: str, columns_to_drop):
    data_X = pd.read_csv(data_X_path)
    data_X = drop_columns_with_lowest_correlation(data_X, columns_to_drop)
    data_y = pd.read_csv(data_y_path)
    data_y = data_y.head(int(len(data_X))).copy()
    data_y = data_y.drop(["0"], axis=1)
    data_y.head()
    X_train, X_test = train_test_split(data_X, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(data_y, test_size=0.2, random_state=42)
    results = linear_regression_test(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    X_valid = pd.read_csv(data_x_valid_path)
    y_valid = pd.read_csv(data_y_valid_path)

    model = results[0]
    y_pred = model.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)  # RMSE
    r2 = r2_score(y_valid, y_pred)  # R²

    print(f"Среднеквадратичная ошибка (RMSE): {rmse}")
    print(f"Коэффициент детерминации (R²): {r2}")

    metrics = {'R²': r2, 'RMSE': rmse}

    return model, metrics

def save_model(model):
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        
def save_metrics(metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open('metrics.txt', 'w') as metrics_file:
        metrics_file.write(f"Timestamp: {timestamp}\n")
        for key, value in metrics.items():
            metrics_file.write(f"{key}: {value}\n")

def main():

    data_X, data_X_path = create_new_df("df", data_X)
    data_X, columns_to_drop = drop_columns_with_lowest_correlation(data_X)
    data_y, data_y_path = load_target()
    data_val, data_val_path = load_data_val(columns_to_drop)
    target_val, target_val_path = load_target_val()

    model, metrics = pipeline(data_X_path=data_X_path, data_y_path=data_y_path, data_x_valid_path=data_val_path, data_y_valid_path=target_val_path, columns_to_drop=columns_to_drop)

    save_metrics(metrics)
    save_model(model)


if __name__ == "__main__":
    main()