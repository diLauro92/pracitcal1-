import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


def load_data (file_path):
    # Загружает данные из CSV-файла и возвращает DataFrame
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден!")
        return None
    
    try: 
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None
    
def preprocess_data (data):
    # Разделяет данные на признаки (X) и целевую переменную (y)
    if 'Price' not in data.columns:
        print("Ошибка: отсутствует целевая переменная 'Price'")
        return None
    
    X = data.drop(columns=['Price'])
    y = data['Price']

    # Определяем числовые и категориальные признаки
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    return X, y, num_features, cat_features

def split_data (X, y, test_size=0.2, random_state=42):
    # Разделяет данные на обучающую и тестовую выборки
    return train_test_split (X, y, test_size=test_size, random_state=random_state) 

def save_to_csv(X_train, X_test, y_train, y_test, output_dir="output"):
    # Сохраняет выборки в файлы CSV
    os.makedirs(output_dir, exist_ok=True)

    for name, data in zip(["X_train", "X_test", "y_train", "y_test"], [X_train, X_test, y_train, y_test]):
        data.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

def get_transformers():
    # Создает и возвращает пайплайны для обработки числовых и категориальных данных
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return num_transformer, cat_transformer

def get_preprocessor(num_transformer, cat_transformer, num_features, cat_features):
    # Создает ColumnTransformer, объединяя числовые и категориальные преобразования
    return ColumnTransformer([
        ('num', num_transformer, num_features),  
        ('cat', cat_transformer, cat_features)  
    ])

def get_pipeline(preprocessor):
    # Создает общий pipeline для обработки данных и обучения модели
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5))
    ])

def save_pipeline(pipeline, output_path="output/laptop_price_model.pkl"):
    # Сохраняет обученный pipeline в файл
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Pipeline сохранен в {output_path}")

def main (file_path="content/laptop_price.csv"):
    # Выполняем все шаги
    data = load_data(file_path)
    if data is None:
        return None
    
    processed = preprocess_data(data)
    if processed is None:
        return None
    
    X, y, num_features, cat_features = processed
    X_train, X_test, y_train, y_test = split_data(X, y)

    save_to_csv(X_train, X_test, y_train, y_test)

    num_transformer, cat_transformer = get_transformers()
    preprocessor = get_preprocessor(num_transformer, cat_transformer, num_features, cat_features)

    pipeline = get_pipeline(preprocessor)
    pipeline.fit(X_train, y_train)
    save_pipeline(pipeline)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_features": num_features,
        "cat_features": cat_features,
        "preprocessor": preprocessor,
        "pipeline": pipeline
    }


if __name__ == "__main__":
    results = main()

    if results:
        X_train = results["X_train"]
        X_test = results["X_test"]
        y_train = results["y_train"]
        y_test = results["y_test"]
        preprocessor = results["preprocessor"]
        pipeline = results["pipeline"]
        num_features = results["num_features"]
        cat_features = results["cat_features"]
else:
    print("Ошибка: `main()` возвращает None")




