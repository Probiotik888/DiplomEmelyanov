import pandas as pd
from textblob import TextBlob
from deep_translator import GoogleTranslator
from utils import create_graphs
import chardet  # Новый импорт


def analyze_reviews(csv_file, db, Review):
    # Сначала определяем кодировку файла
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']

    # Теперь читаем CSV с правильной кодировкой
    df = pd.read_csv(csv_file, encoding=encoding)

    sentiments = []

    for review in df['Отзыв']:
        try:
            # Если отзыв пустой, пропускаем его
            if not review or isinstance(review, float) and pd.isna(review):
                sentiments.append('Нет отзыва')
                continue

            # Печать текста отзыва для отладки
            print(f"Обрабатываем отзыв: {review}")

            # Перевод отзыва на английский
            translated = GoogleTranslator(source='auto', target='en').translate(review)
            print(f"Переведенный текст: {translated}")  # Вывод переведенного текста для отладки

            # Анализ тональности
            blob = TextBlob(translated)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                sentiment = 'Положительный'
            elif polarity < -0.1:
                sentiment = 'Отрицательный'
            else:
                sentiment = 'Нейтральный'
        except Exception as e:
            # Печать ошибки для диагностики
            print(f"Ошибка при анализе отзыва: {review}")
            print(f"Ошибка: {e}")
            sentiment = 'Ошибка анализа'

        sentiments.append(sentiment)

    df['Тональность'] = sentiments

    return df
