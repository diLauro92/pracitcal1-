from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import joblib
from io import BytesIO
import os

# Создаем FastAPI приложение
app = FastAPI(title="Laptop Price Prediction API")

# Загружаем модель
MODEL_PATH = "output/laptop_price_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена в {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

@app.post("/predict/")
async def predict(file: UploadFile = File()):
    # Эндпоинт для предсказания, принимает CSV-файл и возвращает предсказания
    try:
        # Читаем загруженный файл в DataFrame
        content = await file.read()
        df = pd.read_csv(BytesIO(content))

        # Проверяем, что файл не пустой
        if df.empty:
            raise HTTPException(status_code=400, detail="Файл пуст!")

        # Делаем предсказание
        predictions = model.predict(df)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")


@app.get("/")
def home():
    return {"message": "API работает"}
