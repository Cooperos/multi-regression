# multi-regressor
Репозиторий посвещен разработанной на чемпионате REASkills 2025 для решения задачи регрессии

# Quick Start
* Клонируйте репозиторий `git clone https://github.com/Cooperos/multi-regression`
* Скачайте данные для обучения в директорию Data *тут должна быть ссылка*
* Запустите `data_processor.py` локально или в Docker `docker compose up --build` для обучения модели и получения предсказаний

# Модели репозитория
* XGBoost
* RandomForest
* LinearRegression

# Интерфейсы репозитория
* Flask API - `C2_M6_api.py`
* Telegram Bot - `C2_M6_bot.py`

> Необходимые для запуска переменные окружения указаны в `env.example`

# Принцип работы системы
Работа системы делится на этапы:
1. Загрузка данных
2. Аугментация данных
3. Обучение модели регрессии на данных
4. Предсказание на валидационной выборке

## Описание Flask API

##### POST /upload_file
Загрузка .csv файла на сервер
```json
>>> file: File.csv (multipart/form-data)

<<< {"message": <str>}
```

##### POST /preprocess_data
Предобработка данных
```json
>>> args: "filename"

<<< {"message": <str>}
```

##### GET /list_models
Получить список доступных моделей
```json
<<< {"models": [str, str, ...]}
```

##### POST /select_model
Выбрать модель для работы
```json
>>> args: "model"

<<< {"message": str}
```

##### GET /predict
Получить предсказание
```json
>>> args: "filename", "model"

<<< file: File.csv (multipart/form-data)
```

> Работа Telegram бота зависит от FlaskAPI
