from flask import Flask, render_template, make_response, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sentiment_analysis import analyze_reviews
from utils import create_graphs
from datetime import datetime
from weasyprint import HTML
import tempfile
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Секретный ключ для сессий

# Настройка базы данных
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Инициализация
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Модель пользователя
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)


# Модель для отзывов
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product = db.Column(db.String(100), nullable=False)
    review = db.Column(db.String(500), nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f"Review('{self.product}', '{self.sentiment}')"


# Загрузка пользователя
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Создание таблиц
with app.app_context():
    db.create_all()


@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploaded_reviews.csv')
            file.save(filepath)

            df = analyze_reviews(filepath, db, Review)

            for index, row in df.iterrows():
                existing_review = Review.query.filter_by(product=row['Товар'], review=row['Отзыв']).first()
                if not existing_review:
                    review = Review(product=row['Товар'], review=row['Отзыв'], sentiment=row['Тональность'])
                    db.session.add(review)

            db.session.commit()
            create_graphs(db, Review)
            reviews = Review.query.all()
            return render_template('report.html', reviews=reviews)

        return 'Ошибка: файл не загружен'

    reviews = Review.query.all()
    return render_template('report.html', reviews=reviews)


@app.route('/clear', methods=['POST'])
@login_required
def clear():
    try:
        db.session.query(Review).delete()
        db.session.commit()

        graphs_directory = os.path.join(app.root_path, 'static', 'graphs')
        for graph in os.listdir(graphs_directory):
            graph_path = os.path.join(graphs_directory, graph)
            if os.path.isfile(graph_path):
                os.remove(graph_path)

        message = "Таблица и диаграммы успешно очищены."
    except Exception as e:
        db.session.rollback()
        message = f"Ошибка при очистке таблицы: {e}"

    return render_template('index.html', message=message)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template('register.html', message="Пароли не совпадают.")

        if User.query.filter_by(username=username).first():
            return render_template('register.html', message="Имя пользователя уже занято.")

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = 'remember' in request.form

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=remember)
            return redirect(url_for('index'))

        return render_template('login.html', message="Неверное имя пользователя или пароль.")

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/download-pdf')
@login_required
def download_pdf():
    reviews = Review.query.all()  # загружаем отзывы из БД
    rendered = render_template("report_pdf.html", reviews=reviews, current_date=datetime.now().strftime("%d.%m.%Y"))

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        HTML(string=rendered, base_url=request.url_root).write_pdf(tmp_file.name)
        tmp_file.seek(0)
        pdf_data = tmp_file.read()

    response = make_response(pdf_data)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    return response


if __name__ == '__main__':
    app.run(debug=True)
