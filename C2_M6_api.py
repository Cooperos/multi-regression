from flask import Flask, request, jsonify, send_file, g, session
from flask_cors import CORS
import os
from typing import Final
import dask.dataframe as dd
import gc
import pandas as pd
import time
import pickle
import pandas as pd
import xgboost as xgb


app = Flask("my_api")
CORS(app, origins="*")
app.config["UPLOAD_FOLDER"] = "uploads"
DATA_DIR: Final[str] = "uploads"
app.secret_key = "fdsfsdf"
app.config['SESSION_PERMANENT'] = True
##########################################################
######              system part                      #####
##########################################################


def dataframe_loader(path: str) -> dd.DataFrame:
    df = dd.read_csv(path, engine="c", on_bad_lines='skip', )
    df = df.astype('float32') # изначальный тип данных датасета был float64, уменьшаем т.к значения не выходят за пределы размерности этого типа
                            # c float16 уже появляются ошибки о переполнении
    return df

async def load_datasets(name: str): # загружаем данные из каждого файла поочередно и выполняем манипуляции с ними

    df = dataframe_loader(f"{name}")
    print(f"{name} loaded, making describe...")
    yield df
    gc.collect()

async def create_new_df(name: str):
    df_list = []  # Список для хранения частей DataFrame
    async for df_part in load_datasets(name):
        df_list.append(df_part.compute())
        print("Добавлен df в list")

    # Объединяем все части в один DataFrame
    df = pd.concat(df_list, axis=0, join='inner')
    print("добавлен df в df")
    return df

def drop_columns_with_lowest_correlation(df, n=10, columns_to_drop={'26', '5', '27', '6', '12', '3', '19', '25', '16', '2', '17', '7', '0', '11', '21', '14', '10'}):
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


async def preprocess(filename: str) -> str:
    df = await create_new_df(os.path.join(DATA_DIR, filename))
    columns_to_drop = {'26', '5', '27', '6', '12', '3', '19', '25', '16', '2', '17', '7', '0', '11', '21', '14', '10'}
    # df, columns_to_drop = drop_columns_with_lowest_correlation(df, columns_to_drop=columns_to_drop)
    path = os.path.join(DATA_DIR, f"file.csv")
    df.to_csv(path)
    return path

def load_model(model_path: str):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def xgboost_predict(file_path: str, model_path: str = "xgboost_model.pkl") -> pd.DataFrame:
    xg_model = load_model(model_path=model_path)
    X = pd.read_csv(file_path)
    X = X.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    X = xgb.DMatrix(X)
    y_pred = xg_model.predict(X)
    return y_pred


def random_forest_predict(file_path: str, model_path: str = "rf_model.pkl") -> pd.DataFrame:
    rf_model = load_model(model_path=model_path)
    X = pd.read_csv(file_path)
    X = X.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    y_pred = rf_model.predict(X)
    return y_pred

def linear_regression_predict(file_path: str, model_path: str = "lr_model.pkl") -> pd.DataFrame:
    lr_model = load_model(model_path=model_path)
    X = pd.read_csv(file_path)
    X = X.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    y_pred = lr_model.predict(X)
    return y_pred

def predict_data(filename: str, MODEL_SELECTED: str)  -> str:
    result = None
    print("MODEL",MODEL_SELECTED)
    if MODEL_SELECTED == "XGBoost":
        result = xgboost_predict(filename)
    elif MODEL_SELECTED == "RandomForest":
        result = random_forest_predict(filename)
    elif MODEL_SELECTED == "LinearRegression":
        result = linear_regression_predict(filename)

    path = os.path.join(DATA_DIR, f"prediction_result.csv")
    result = pd.DataFrame(result)
    result.to_csv(path)
    print(result)
    return path

##########################################################
######                 API part                      #####
##########################################################

@app.route("/upload_file", methods=["POST"])
def upload_file():
    """
    file (multipart)
    """
    try:
        request._load_form_data()
        
        if not "file" in request.files:
            return jsonify({"message": "no file in files"}), 400
        
        file = request.files["file"]
        print("файл найден")
        
        # if not file.name.endswith(".csv"):
        #     print(file.name)
        #     return jsonify({"message": "Only CSV files are acceptable"}), 400
        
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.name + ".csv"))
        print("файл сохранен")
    except Exception as e:
        print("error", e)
    return jsonify({"message": "File saved on server"}), 200

@app.route("/preprocess_data", methods=["POST"])
async def preprocess_data():
    if not "filename" in request.args:
        return jsonify({"message": "no file name in request arguments"})
    
    filename = request.args.get("filename")

    path = await preprocess(filename)
    print("данные обработаны")

    if not path: 
        return jsonify({"message": "server error in preprocessing"}), 500
    
    return jsonify({"message": "data preprocessed", "path": path}), 200

@app.route("/list_models", methods=["GET"])
def list_models():
    return jsonify({"models": ["XGBoost", "RandomForest", "LinearRegression"]}), 200

@app.route("/select_model", methods=["POST"])
def select_model():
    if not "model" in request.args:
        return jsonify({"message": "model not selected"}), 400
    session["model_name"] = request.args.get("model")
    print("модель выбрана", session["model_name"])

    return jsonify({"message": f"model selected as {session['model_name']}"}), 200

@app.route('/predict', methods=["GET"])
def predict():
    if not "filename" in request.args:
        return jsonify({"message": "no filename in request args"}), 400
    
    filename = request.args.get("filename")
    model_name = request.args.get("model")

    result_file_path = predict_data(os.path.join(DATA_DIR, filename), model_name)
    print("данные  получены")

    return send_file(result_file_path, as_attachment=True, download_name="prediction_result.csv")

if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    app.run(host="0.0.0.0", port=7777, debug=True)