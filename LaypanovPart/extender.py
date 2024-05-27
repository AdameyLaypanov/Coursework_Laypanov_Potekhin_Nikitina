from datetime import datetime
import sqlite3
import pandas as pd

# Подключение к существующей базе данных SQLite
db_path = './../messages.db'  # Укажите путь к вашей базе данных
conn = sqlite3.connect(db_path)

# Получение списка таблиц в базе данных
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(tables_query, conn)
tables_list = tables['name'].tolist()

# Загрузка данных из всех таблиц в DataFrame
sql_dataframes = {table: pd.read_sql(f"SELECT * FROM {table};", conn) for table in tables_list}

# Отображение первых нескольких строк каждой таблицы
sql_dataframes_sample = {table: df.head() for table, df in sql_dataframes.items()}
sql_dataframes_sample
# Пути к CSV файлам
csv_files = {
    'etf': './../etf.csv',
    'alpha': './../alpha.csv',
    'ton': './../ton.csv'
}

# Загрузка данных из CSV файлов в DataFrame
csv_dataframes = {name: pd.read_csv(path, delimiter=';') for name, path in csv_files.items()}

# Отображение первых нескольких строк каждого CSV файла
csv_dataframes_sample = {name: df.head() for name, df in csv_dataframes.items()}
csv_dataframes_sample
for n in csv_dataframes:
    csv_dataframes_sample[n].columns = ["date", "q", "p", "standardized_date"]
    csv_dataframes_sample[n]["standardized_date"]=csv_dataframes_sample[n]["date"]


import json
def message_date(r):
    return datetime.fromtimestamp(json.loads(r)["date"]).strftime('%d.%m.%Y')

sql_dataframes["messages"]['standardized_date']=sql_dataframes["messages"]["message_object"].apply(message_date)


def news_date(e):
    return e.split(" ")[0]


def update_database(conn, table_name, df):
    df.to_sql(table_name, conn, if_exists='replace', index=False)

# Обновление таблиц в базе данных
update_database(conn, "messages", sql_dataframes["messages"])

for file_name, df in csv_dataframes.items():
    update_database(conn, file_name, df)



# Функция для преобразования даты в стандартизированный формат
def standardize_date(date_str):
    try:
        return datetime.strptime(date_str, '%d.%m.%Y')
    except ValueError:
        return None

conn.close()
# Применение функции для всех DataFrame
sql_dataframes["news"]['standardized_date']=sql_dataframes["news"]["date"].apply(news_date)

# Функция для обновления базы данных
update_database(conn, "news", sql_dataframes["news"])


# Закрытие соединения с базой данных
conn.close()
# Добавление данных из CSV файлов в базу данных
