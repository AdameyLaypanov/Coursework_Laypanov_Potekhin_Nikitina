import sqlite3
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from spacy.pipeline import EntityRuler
from spacy.language import Language
from transformers import pipeline
from langdetect import detect
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Загрузка словарей NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Загрузка модели spaCy для русского языка
nlp = spacy.load('ru_core_news_md')

# Подключение к базе данных SQLite и загрузка данных
db_path = './../messages.db'
conn = sqlite3.connect(db_path)
news_df = pd.read_sql("SELECT * FROM news", conn)

# Проверка загруженных данных
print(news_df.head())

# Список ключевых слов для фильтрации финансовых новостей
financial_keywords = [
    "рынок", "акции", "облигации", "биржа", "цена", "торговля", "инфляция",
    "процентная ставка", "ВВП", "доходы", "инвестиции", "прибыль", "убыток",
    "выручка", "дивиденды", "прогноз", "экономика", "финансы", "банк", "криптовалюта"
]


# Функция для фильтрации новостей по ключевым словам
def filter_financial_news(df, text_column, keywords):
    pattern = re.compile('|'.join(keywords), re.IGNORECASE)
    filtered_df = df[df[text_column].str.contains(pattern)]
    return filtered_df


# Применение фильтрации к новостям
filtered_news_df = filter_financial_news(news_df, 'text', financial_keywords)

# Проверка отфильтрованных данных
print(filtered_news_df[['text']].head())


# Сентимент-анализ с использованием TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


# Применение сентимент-анализа
filtered_news_df['sentiment'] = filtered_news_df['text'].apply(analyze_sentiment)

# Проверка результатов сентимент-анализа
print(filtered_news_df[['text', 'sentiment']].head())

# Получение русских стоп-слов
stop_words = set(stopwords.words('russian'))


# Функция для предобработки текста
def preprocess_text(text):
    # Удаление пунктуации и цифр, приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    # Токенизация
    words = word_tokenize(text)

    # Удаление стоп-слов
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)


# Применение предобработки к текстам новостей
filtered_news_df['clean_text'] = filtered_news_df['text'].apply(preprocess_text)


# Выделение сущностей с использованием spaCy
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


filtered_news_df['entities'] = filtered_news_df['text'].apply(extract_entities)

# Проверка результатов выделения сущностей
print(filtered_news_df[['text', 'entities']].head())

# Применение TF-IDF
vectorizer = TfidfVectorizer(max_features=10)  # Максимальное количество тэгов для каждого текста
tfidf_matrix = vectorizer.fit_transform(filtered_news_df['clean_text'])

# Получение значимых тэгов для каждого текста
feature_names = vectorizer.get_feature_names_out()
tags_list = []
for doc in tfidf_matrix:
    feature_index = doc.nonzero()[1]
    tfidf_scores = zip(feature_index, [doc[0, x] for x in feature_index])
    tags = [feature_names[i] for i, score in tfidf_scores]
    tags_list.append(', '.join(tags))

# Добавление тэгов в исходный DataFrame новостей
filtered_news_df['tags'] = tags_list

# Проверка результатов
print(filtered_news_df[['text', 'tags']].head())

# Запись результатов обратно в базу данных
filtered_news_df.to_sql('financial_news_analysis', conn, if_exists='replace', index=False)

print("Анализ финансовых новостей успешно завершен и результаты добавлены в таблицу financial_news_analysis.")
