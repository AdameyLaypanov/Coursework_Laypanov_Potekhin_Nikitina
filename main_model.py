import sqlite3

import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Connect to the SQLite database
db_path = './../messages.db'
conn = sqlite3.connect(db_path)

# Load data from each table
messages_df = pd.read_sql("SELECT * FROM messages", conn)
ethereum_search_frequencies_df = pd.read_sql("SELECT * FROM Ethereum_serch_frequencies", conn)
alpha_bank_search_frequencies_df = pd.read_sql("SELECT * FROM alpha_Bank_serch_frequencies", conn)
toncoin_search_frequencies_df = pd.read_sql("SELECT * FROM Toncoin_serch_frequencies", conn)
news_df = pd.read_sql("SELECT * FROM news", conn)
toncoin_df = pd.read_sql("SELECT * FROM Toncoin", conn)
ethereum_df = pd.read_sql("SELECT * FROM Ethereum", conn)
alphabank_df = pd.read_sql("SELECT * FROM Alphabank", conn)

# Convert 'absolute_queries' columns to numeric
ethereum_search_frequencies_df['absolute_queries'] = (ethereum_search_frequencies_df['absolute_queries'])
alpha_bank_search_frequencies_df['absolute_queries'] = (alpha_bank_search_frequencies_df['absolute_queries'])
toncoin_search_frequencies_df['absolute_queries'] = (toncoin_search_frequencies_df['absolute_queries'])

# Merge datasets based on 'standardized_date'
merged_df = ethereum_df.merge(
    ethereum_search_frequencies_df[['standardized_date', 'absolute_queries']],
    on='standardized_date', how='left', suffixes=('', '_eth_search')
).merge(
    alpha_bank_search_frequencies_df[['standardized_date', 'absolute_queries']],
    on='standardized_date', how='left', suffixes=('', '_alpha_search')
).merge(
    toncoin_search_frequencies_df[['standardized_date', 'absolute_queries']],
    on='standardized_date', how='left', suffixes=('', '_ton_search')
).merge(
    toncoin_df[['standardized_date', 'close']],
    on='standardized_date', how='left', suffixes=('', '_toncoin')
).merge(
    alphabank_df[['standardized_date', 'closePrice']],
    on='standardized_date', how='left', suffixes=('', '_alphabank')
)

# Define a function to calculate sentiment from a message
def calculate_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# Apply sentiment analysis on messages
messages_df = messages_df[messages_df['message'].notna()]
messages_df[['polarity', 'subjectivity']] = messages_df['message'].apply(lambda x: pd.Series(calculate_sentiment(x)))

# Aggregate daily sentiment scores
daily_sentiment = messages_df.groupby('standardized_date')[['polarity', 'subjectivity']].mean().reset_index()

# Merge daily sentiment scores into the merged_df
merged_df = merged_df.merge(daily_sentiment, on='standardized_date', how='left')

# Apply sentiment analysis on news
news_df = news_df[news_df['text'].notna()]
news_df[['polarity', 'subjectivity']] = news_df['text'].apply(lambda x: pd.Series(calculate_sentiment(x)))

# Aggregate daily sentiment scores from news
daily_news_sentiment = news_df.groupby('standardized_date')[['polarity', 'subjectivity']].mean().reset_index()

# Merge daily news sentiment scores into the merged_df
merged_df = merged_df.merge(daily_news_sentiment, on='standardized_date', how='left', suffixes=('_msg', '_news'))

# Handle missing values by filling or interpolating them
merged_df.fillna(method='ffill', inplace=True)
merged_df.interpolate(method='linear', inplace=True)

# Function to create lagged features
def create_lagged_features(df, columns, lags):
    for column in columns:
        for lag in lags:
            df[f'{column}_lag{lag}'] = df[column].shift(lag)
    return df

# Columns to create lagged features for
columns_to_lag = ['absolute_queries', 'absolute_queries_alpha_search', 'absolute_queries_ton_search',
                  'close', 'close_toncoin', 'closePrice', 'polarity_msg', 'subjectivity_msg',
                  'polarity_news', 'subjectivity_news']

# Lags to create
lags = [1, 2, 3, 5, 7]

# Create lagged features
merged_df_with_lags = create_lagged_features(merged_df, columns_to_lag, lags)
merged_df_with_lags.dropna(inplace=True)

# Calculate daily aggregates for message sentiment and count the number of messages per day
daily_message_sentiment = messages_df.groupby('standardized_date').agg({
    'polarity': ['mean', 'sum'],
    'subjectivity': ['mean', 'sum'],
    'message': 'count'
}).reset_index()

# Flatten the column names
daily_message_sentiment.columns = ['standardized_date', 'polarity_mean_msg', 'polarity_sum_msg',
                                   'subjectivity_mean_msg', 'subjectivity_sum_msg', 'message_count']

# Merge the daily message sentiment and count into the merged_df_with_lags
merged_df_with_social = merged_df_with_lags.merge(daily_message_sentiment, on='standardized_date', how='left')
merged_df_with_social.fillna(0, inplace=True)

# Define the target variable and feature variables
target = 'close'
features = merged_df_with_social.columns.difference(['standardized_date', 'timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'name', 'timestamp', target])
print(merged_df_with_social.head())
# Split the data into training and testing sets
X = merged_df_with_social[features]
y = merged_df_with_social[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Function to manually tune hyperparameters for Random Forest Regressor
def manual_rf_tuning(X_train, y_train, X_test, y_test):
    results = []
    for n_estimators in [100, 200]:
        for max_depth in [10, 20, None]:
            for min_samples_split in [2, 5]:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(
                    f"Random Forest - MAE: {mae}, MSE: {mse}, R²: {r2} [estimators:{n_estimators}, depth:{max_depth}, split{min_samples_split}]")
                results.append((n_estimators, max_depth, min_samples_split, mae, mse, r2))
    return results

# Perform manual hyperparameter tuning for Random Forest
rf_results = manual_rf_tuning(X_train, y_train, X_test, y_test)

# Find the best combination for Random Forest based on R² score
best_rf_result = sorted(rf_results, key=lambda x: x[5], reverse=True)[0]
print("Best Random Forest Parameters and Score:", best_rf_result)

# Function to manually tune hyperparameters for Gradient Boosting Regressor
def manual_gbr_tuning(X_train, y_train, X_test, y_test):
    results = []
    for n_estimators in [100, 200]:
        for learning_rate in [0.01, 0.1]:
            for max_depth in [3, 5]:
                model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(
                    f"Gradient Boosting - MAE: {mae}, MSE: {mse}, R²: {r2} [estimators: {n_estimators}, rate:{learning_rate}, depth{max_depth}]")

                results.append((n_estimators, learning_rate, max_depth, mae, mse, r2))
    return results

# Perform manual hyperparameter tuning for Gradient Boosting
gbr_results = manual_gbr_tuning(X_train, y_train, X_test, y_test)

# Find the best combination for Gradient Boosting based on R² score
best_gbr_result = sorted(gbr_results, key=lambda x: x[5], reverse=True)[0]
print("Best Gradient Boosting Parameters and Score:", best_gbr_result)



# Функция для создания признаков с использованием TF-IDF
def create_tfidf_features(df, column_name, max_features=100):
    tfidf = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df[column_name].fillna(''))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df['standardized_date'] = df['standardized_date'].values
    return tfidf_df

# Создание признаков из текстов новостей
tfidf_features = create_tfidf_features(news_df, 'text')
merged_df_with_tfidf = merged_df_with_social.merge(tfidf_features, on='standardized_date', how='left')

# Обработка пропущенных значений
merged_df_with_tfidf.fillna(0, inplace=True)

# Определение целевой переменной и признаков
target = 'close'
features = merged_df_with_tfidf.columns.difference(['standardized_date', 'timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'name', 'timestamp', target])

# Разделение данных на обучающие и тестовые наборы
X = merged_df_with_tfidf[features]
y = merged_df_with_tfidf[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Функция для ручной настройки гиперпараметров для Random Forest Regressor
def manual_rf_tuning(X_train, y_train, X_test, y_test):
    results = []
    for n_estimators in [100, 200]:
        for max_depth in [10, 20, None]:
            for min_samples_split in [2, 5]:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"Random Forest - MAE: {mae}, MSE: {mse}, R²: {r2} [estimators:{n_estimators}, depth:{max_depth}, split{min_samples_split}]")
                results.append((n_estimators, max_depth, min_samples_split, mae, mse, r2))
    return results

# Функция для ручной настройки гиперпараметров для Gradient Boosting Regressor
def manual_gbr_tuning(X_train, y_train, X_test, y_test):
    results = []
    for n_estimators in [100, 200]:
        for learning_rate in [0.01, 0.1]:
            for max_depth in [3, 5]:
                model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"Gradient Boosting - MAE: {mae}, MSE: {mse}, R²: {r2} [estimators: {n_estimators}, rate:{learning_rate}, depth{max_depth}]")
                results.append((n_estimators, learning_rate, max_depth, mae, mse, r2))
    return results

# Ручная настройка гиперпараметров для Random Forest
rf_results = manual_rf_tuning(X_train, y_train, X_test, y_test)

# Поиск лучшей комбинации для Random Forest на основе R² score
best_rf_result = sorted(rf_results, key=lambda x: x[5], reverse=True)[0]
print("Best Random Forest Parameters and Score:", best_rf_result)

# Ручная настройка гиперпараметров для Gradient Boosting
gbr_results = manual_gbr_tuning(X_train, y_train, X_test, y_test)

# Поиск лучшей комбинации для Gradient Boosting на основе R² score
best_gbr_result = sorted(gbr_results, key=lambda x: x[5], reverse=True)[0]
print("Best Gradient Boosting Parameters and Score:", best_gbr_result)

# Визуализация важности признаков для лучших моделей
def plot_feature_importances(model, model_name, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Топ 20 признаков
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importances - {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

# Обучение лучших моделей
best_rf_model = RandomForestRegressor(n_estimators=best_rf_result[0], max_depth=best_rf_result[1], min_samples_split=best_rf_result[2], random_state=42)
best_rf_model.fit(X_train, y_train)

best_gbr_model = GradientBoostingRegressor(n_estimators=best_gbr_result[0], learning_rate=best_gbr_result[1], max_depth=best_gbr_result[2], random_state=42)
best_gbr_model.fit(X_train, y_train)

# Визуализация важности признаков
plot_feature_importances(best_rf_model, "Random Forest", features)
plot_feature_importances(best_gbr_model, "Gradient Boosting", features)

# Предсказания и визуализация точности прогнозов
def plot_predictions(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

plot_predictions(best_rf_model, X_test, y_test, "Random Forest")
plot_predictions(best_gbr_model, X_test, y_test, "Gradient Boosting")

# Оценка лучших моделей
rf_mae = mean_absolute_error(y_test, best_rf_model.predict(X_test))
rf_mse = mean_squared_error(y_test, best_rf_model.predict(X_test))
rf_r2 = r2_score(y_test, best_rf_model.predict(X_test))

gbr_mae = mean_absolute_error(y_test, best_gbr_model.predict(X_test))
gbr_mse = mean_squared_error(y_test, best_gbr_model.predict(X_test))
gbr_r2 = r2_score(y_test, best_gbr_model.predict(X_test))

print(f"Random Forest - MAE: {rf_mae}, MSE: {rf_mse}, R²: {rf_r2}")
print(f"Gradient Boosting - MAE: {gbr_mae}, MSE: {gbr_mse}, R²: {gbr_r2}")
