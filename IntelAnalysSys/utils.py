import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import string

matplotlib.use('Agg')  # Отключить GUI для matplotlib

nltk.download('stopwords')
russian_stopwords = set(stopwords.words("russian"))


def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in russian_stopwords]
    return ' '.join(words)


def create_semantic_graphs(df):
    graphs_path = 'static/graphs'

    all_reviews = ' '.join(df['Отзыв'].dropna().astype(str))
    processed_text = preprocess_text(all_reviews)

    # 🌥 Облако слов
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(processed_text)
    cloud_path = os.path.join(graphs_path, 'wordcloud.png')
    wordcloud.to_file(cloud_path)
    print(f"Облако слов сохранено по пути: {cloud_path}")

    # 📊 Частотные фразы
    vectorizer = CountVectorizer(ngram_range=(2, 3), max_features=20)
    X = vectorizer.fit_transform([processed_text])
    phrases = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)

    phrase_df = pd.DataFrame({'Фраза': phrases, 'Частота': counts}).sort_values(by='Частота', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(phrase_df['Фраза'], phrase_df['Частота'], color='skyblue')
    plt.title("Наиболее частотные фразы")
    plt.xlabel("Частота")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    phrase_path = os.path.join(graphs_path, 'frequent_phrases.png')
    plt.savefig(phrase_path)
    plt.close()
    print(f"Диаграмма частотных фраз сохранена по пути: {phrase_path}")


def create_graphs(db, Review):
    graphs_path = 'static/graphs'

    # Удаляем старые графики
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)
    else:
        old_files = glob.glob(os.path.join(graphs_path, '*'))
        for file in old_files:
            os.remove(file)

    # 📥 Загружаем ВСЕ данные из базы через SQLAlchemy
    reviews = Review.query.all()

    # Преобразуем данные в DataFrame для удобства работы
    df = pd.DataFrame([(r.product, r.review, r.sentiment) for r in reviews], columns=['Товар', 'Отзыв', 'Тональность'])

    # 📊 Круговая диаграмма тональностей
    sentiment_counts = df['Тональность'].value_counts()
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff9999', '#99ff99'],
        wedgeprops={'edgecolor': 'black'},
        textprops={'fontsize': 12}
    )
    plt.ylabel('')
    plt.title('Распределение тональностей отзывов', fontsize=16)
    plt.tight_layout()

    pie_path = os.path.join(graphs_path, 'sentiment_pie.png')
    plt.savefig(pie_path)
    plt.close()
    print(f"График круговой диаграммы сохранен по пути: {pie_path}")

    # 📊 Столбчатая диаграмма проблемных товаров
    negative_products = df[df['Тональность'] == 'Отрицательный']['Товар'].value_counts()
    if not negative_products.empty:
        plt.figure(figsize=(8, 6))
        negative_products.plot(kind='bar', color='#ff6666', edgecolor='black')
        plt.title('Товары с негативными отзывами', fontsize=16)
        plt.xlabel('Товар', fontsize=14)
        plt.ylabel('Количество негативных отзывов', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        bar_path = os.path.join(graphs_path, 'negative_products.png')
        plt.savefig(bar_path)
        plt.close()
        print(f"График столбчатой диаграммы сохранен по пути: {bar_path}")

        create_semantic_graphs(df)
