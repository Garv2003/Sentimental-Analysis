from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
import re

app = Flask(__name__)


def load_my_model(model_path):
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except OSError as e:
        print(f"Error loading model: {e}")
        return None


def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)


def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem


def to_lower(text):
    return text.lower()


def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]


def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])


def clean_text(text):
    text = clean(text)
    text = is_special(text)
    text = to_lower(text)
    text = rem_stopwords(text)
    text = stem_txt(text)
    return text


model = load_my_model('./model/Sentimental_Analysis.keras')

max_sequence_length = 100


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.form['user_input'] == '':
        return "<p id='prediction' class='flex items-center justify-center w-96 p-4 mt-4 bg-white shadow-md rounded-lg text-red-800 font-bold'>Please enter a valid input</p>"

    try:
        user_input = request.form['user_input']
        user_input = clean_text(user_input)
        tokenizer = Tokenizer()
        user_sequences = tokenizer.texts_to_sequences([user_input])
        user_padded = tf.keras.preprocessing.sequence.pad_sequences(
            user_sequences, maxlen=max_sequence_length)
        class_mapping = {
            0: 'Negative',
            1: 'Neutral',
            2: 'Positive'
        }
        user_predictions = model.predict(user_padded)
        print(user_predictions)
        user_pred_classes = np.argmax(user_predictions, axis=1)
        print(user_pred_classes)
        print(f'Predicted Class: {class_mapping[user_pred_classes[0]]}')
        return "<p id='prediction' class='flex items-center justify-center w-96 p-4 mt-4 bg-white shadow-md rounded-lg text-gray-800 font-bold'>" + class_mapping[user_pred_classes[0]].capitalize() + "</p>"

    except Exception as e:
        print(f"Error predicting image: {e}")
        return "<p id='prediction' class='flex items-center justify-center w-96 p-4 mt-5 bg-white shadow-md rounded-lg text-red-800 font-bold'>Error predicting image please try again</p>"


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
