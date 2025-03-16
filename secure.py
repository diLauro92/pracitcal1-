import pandas as pd
import hashlib
import re
from cryptography.fernet import Fernet

# Загружаем данные
file_path = 'content/laptop_price.csv'
try:
    data = pd.read_csv(file_path)
except Exception as e:
    print(f'Fail download: {e}')
    data = None

# Проверяем CSV на уязвимости (CSV Injection) 
# проверяем наличие спецсимволов в тексте
def check_csv_injection (data):
    if data is None:
        print("Данные не найдены")
        return

    dangerous_chars = ('=', '+', '-', '@')
    pattern = re.compile(r'^\s*[' + re.escape(''.join(dangerous_chars)) + ']')

    for col in data.select_dtypes(include=['object']).columns:
        if data[col].astype(str).apply(lambda x: bool(pattern.match(x))).any():
            print(f'Найдены потенциальные CSV-иньекции в столбце {col}')
        else:
            print(f'Столбец {col} безопасен')

check_csv_injection(data)                            

# Фильтрация данных от SQL-иньекций и XSS атак
# Заменяем опасные конструкции на строку '[BLOCKED]'
def clean_input(value):
    sql_keywords = ['SELECT', 'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'UNION', '--']
    xss_patterns = [r'<script.*?>.*?</script>', r'javascript:.*', r'oneerror=.*']

    for keyword in sql_keywords:
        if keyword.lower() in value.lower():
            return '[BLOCKED]'
    
    for pattern in xss_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            return '[BLOCKED]'
    
    return value

data = data.applymap(lambda x: clean_input(str(x)) if isinstance(x, str) else x)
print('Фильтрация данных завершена')

# Хешируем столбец с ценами (SHA-256)
def hash_price (price):
    return hashlib.sha256(str(price).encode()).hexdigest()

data['Price_Hashed'] = data['Price'].apply(hash_price)
print('Столбец с хешированными ценами добавлен')

# Шифруем данные 
# Сгенерируем ключ и зашифруем цену
key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_price(price):
    return cipher.encrypt(str(price).encode()).decode()

def decrypt_price(encrypt_price):
    return cipher.decrypt(encrypt_price.encode()).decode()

data['Price_Encrypted'] = data['Price'].apply(encrypt_price)
print('Зашифрованная цена добавлена')

# Шифруем столбец RAM
def encrypt_ram(RAM_Size):
    return cipher.encrypt(str(RAM_Size).encode()).decode()

data['RAM_Encrypted'] = data['RAM_Size'].apply(encrypt_ram)
print('Зашифрованная RAM добавлена')

# Дешифруем столбец ram
def decrypt_ram(encrypted_ram):
    return cipher.decrypt(encrypted_ram.encode()).decode()

data['RAM_Decrypted'] = data['RAM_Encrypted'].apply(decrypt_ram)
print("Первые 5 расшифрованных значений RAM:")
print(data['RAM_Decrypted'].head(5))

# Сохраняем данные
output_path = "output/Laptop_price_secured.csv"
data.to_csv(output_path, index=False)
print(f'Обработанный файл сохранен: {output_path}')