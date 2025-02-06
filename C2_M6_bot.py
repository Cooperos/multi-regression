import logging
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher import filters
from aiogram.utils import executor
from aiogram.dispatcher.filters.state import State, StatesGroup
import os

API_URL = "http://localhost:7777"  # URL вашего Flask API
API_UPLOAD_FILE = f"{API_URL}/upload_file"
API_PREPROCESS_DATA = f"{API_URL}/preprocess_data"
API_LIST_MODELS = f"{API_URL}/list_models"
API_SELECT_MODEL = f"{API_URL}/select_model"
API_PREDICT = f"{API_URL}/predict"
BOT_FILES = "./bot_files"

if not os.path.exists(BOT_FILES):
    os.makedirs(BOT_FILES, exist_ok=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Создание бота и диспетчера
bot = Bot(token='8003615625:AAFSNR6_jLOvWt-TMBkhy_7iz-JzTIrIOPw')
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Определение состояний
class Form(StatesGroup):
    waiting_for_file = State()
    waiting_for_model = State()
    waiting_for_preprocess = State()
    waiting_for_predict = State()

@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    hello_msg = (
    f"Привет! Я бот для работы с данными. Используйте /upload для загрузки файла.\n"
    f"Доступные команды:\n"
    f"/upload - загрузить файл\n"
    f"/preprocess - начать предобработку файла\n"
    f"/list_models - посмотреть список доступных моделей\n"
    f"/select_model <название модели> - выбрать модель из списка\n"
    f"/predict - начать процесс предсказания результата и получить ответ\n"
    f"\nВыполняйте команды последовательно, а то ничего не получится :)"
)
    await message.answer(hello_msg)

@dp.message_handler(commands='upload')
async def cmd_upload(message: types.Message):
    await message.answer("Пожалуйста, отправьте CSV файл.")
    await Form.waiting_for_file.set()

@dp.message_handler(state=Form.waiting_for_file, content_types=types.ContentType.DOCUMENT)
async def process_file(message: types.Message, state: FSMContext):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    print(file_path)
    # download_path = os.path.join(BOT_FILES, file_path)
    dest = await file.download(destination_dir=BOT_FILES)
    print(f"file saved at {dest}")
    # Скачиваем файл
    # file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_path}"
    # response = requests.get(file_url)

    # Сохраняем файл на сервере
    # with open(message.document.file_name, 'wb') as f:
    #     f.write(response.content)

    # Загружаем файл на Flask API
    try:
        with open(os.path.join(BOT_FILES, "documents/file_0.csv"), 'rb') as f:
            files = {'file': f}
            api_response = requests.post(API_UPLOAD_FILE, files=files)
    except Exception as e:
        print("error ", e)
    if api_response.status_code == 200:
        await message.answer("Файл успешно загружен.")
    else:
        await message.answer(f"Ошибка при загрузке файла: {api_response.content}")

    await state.finish()

@dp.message_handler(commands='preprocess')
async def cmd_preprocess(message: types.Message):
    await message.answer("Введите имя файла для предобработки:")
    await Form.waiting_for_preprocess.set()

@dp.message_handler(state=Form.waiting_for_preprocess)
async def process_preprocess_file(message: types.Message, state: FSMContext):
    filename = message.text
    api_response = requests.post(API_PREPROCESS_DATA, params={"filename": filename})

    if api_response.status_code == 200:
        await message.answer("Данные успешно предобработаны.")
    else:
        await message.answer("Ошибка при предобработке данных.")

    await state.finish()

@dp.message_handler(commands='list_models')
async def cmd_list_models(message: types.Message):
    api_response = requests.get(API_LIST_MODELS)
    models = api_response.json().get("models", [])
    await message.answer("Доступные модели: " + ", ".join(models))

@dp.message_handler(commands='select_model')
async def cmd_select_model(message: types.Message):
    await message.answer("Введите название модели (XGBoost, RandomForest, LinearRegression):")
    await Form.waiting_for_model.set()

@dp.message_handler(state=Form.waiting_for_model)
async def process_select_model(message: types.Message, state: FSMContext):
    model_name = message.text
    api_response = requests.post(API_SELECT_MODEL, params={"model": model_name})

    if api_response.status_code == 200:
        await message.answer(f"Модель выбрана: {model_name}.")
    else:
        await message.answer("Ошибка при выборе модели.")

    await state.finish()

@dp.message_handler(commands='predict')
async def cmd_predict(message: types.Message):
    await message.answer("Введите имя файла для предсказания и модели через пробел:")
    await Form.waiting_for_predict.set()

@dp.message_handler(state=Form.waiting_for_predict)
async def process_predict_file(message: types.Message, state: FSMContext):
    text_data = message.text.split(" ")
    filename = text_data[0]
    model = text_data[1]
    
    # Выполняем запрос к API для получения предсказания
    api_response = requests.get(API_PREDICT, params={"filename": filename, "model": model})

    if api_response.status_code == 200:
        # Сохраняем результат предсказания в файл
        result_file_path = "prediction_result.csv"
        with open(result_file_path, 'wb') as f:
            f.write(api_response.content)
        
        # Отправляем файл пользователю
        with open(result_file_path, 'rb') as f:
            await message.answer_document(f, caption="Предсказание завершено. Результаты сохранены в файл prediction_result.csv.")
            
    else:
        await message.answer("Ошибка при получении предсказания.")

    await state.finish()

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)