# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:28:10 2024

@author: hp
"""

from flask import Flask, request, render_template, jsonify
import joblib
import os
import base64
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd

app = Flask(__name__)

# Load the dataset and train the model
data = pd.read_csv('E:/neural/data.csv', encoding='latin1')
X_train, X_test, y_train, y_test = train_test_split(data['Misspelled'], data['Correct'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1, 3))),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])

pipeline.fit(X_train, y_train)

# Save the model
model_path = 'models/misspelling_correction_model.h5'
joblib.dump(pipeline, model_path)

# Word to image dictionary (replace with your image paths)
word_to_image = {
    'hello': 'C:/Users/hp/Desktop/Deaf Hub/static/hello.JPG',
    'please': 'C:/Users/hp/Desktop/Deaf Hub/static/please.JPG',
    'all done': 'C:/Users/hp/Desktop/Deaf Hub/static/all done.JPG',
    'water': 'C:/Users/hp/Desktop/Deaf Hub/static/water.JPG',
    'eat': 'C:/Users/hp/Desktop/Deaf Hub/static/eat.JPG',
    'hungry': 'C:/Users/hp/Desktop/Deaf Hub/static/hungry.JPG',
    'thank you': 'C:/Users/hp/Desktop/Deaf Hub/static/thank you.JPG',
    'more': 'C:/Users/hp/Desktop/Deaf Hub/static/more.JPG',
    'help': 'C:/Users/hp/Desktop/Deaf Hub/static/help.JPG',
    'don\'t': 'C:/Users/hp/Desktop/Deaf Hub/static/don\'t.JPG',
    'no': 'C:/Users/hp/Desktop/Deaf Hub/static/no.JPG',
    'play': 'C:/Users/hp/Desktop/Deaf Hub/static/play.JPG',
    'toilet': 'C:/Users/hp/Desktop/Deaf Hub/static/toilet.JPG',
    'what': 'C:/Users/hp/Desktop/Deaf Hub/static/what.JPG',
    'when': 'C:/Users/hp/Desktop/Deaf Hub/static/when.JPG',
    'where': 'C:/Users/hp/Desktop/Deaf Hub/static/where.JPG',
    'who': 'C:/Users/hp/Desktop/Deaf Hub/static/who.JPG',
    'why': 'C:/Users/hp/Desktop/Deaf Hub/static/why.JPG',
    'yes': 'C:/Users/hp/Desktop/Deaf Hub/static/yes.JPG',
    'you': 'C:/Users/hp/Desktop/Deaf Hub/static/you.JPG',
    'how are you': 'C:/Users/hp/Desktop/Deaf Hub/static/how are you.gif',
    'I like it': 'C:/Users/hp/Desktop/Deaf Hub/static/i like it.gif',
    'are you okay?': 'C:/Users/hp/Desktop/Deaf Hub/static/are you okay.gif',
    'see you later': 'C:/Users/hp/Desktop/Deaf Hub/static/see you later.gif',
    'i don\'t know': 'C:/Users/hp/Desktop/Deaf Hub/static/i don\'t know.gif'
}

# Load the model
pipeline = joblib.load(model_path)

def get_image_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return None

def get_voice_text():
    """
    This function listens for user's voice input, recognizes it using Google Speech-to-Text,
    and returns the converted text.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak anything:")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speak')
def speak():
    return render_template('speak.html')

@app.route('/type')
def type():
    return render_template('type.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    word = data['word']
    corrected_word = pipeline.predict([word])[0]
    image_path = word_to_image.get(corrected_word)
    image_base64 = get_image_base64(image_path) if image_path else None

    response = {
        'typed_word': word,
        'corrected_word': corrected_word,
        'image': image_base64
    }
    return jsonify(response)

@app.route('/voice', methods=['POST'])
def voice():
    text = get_voice_text()
    if text:
        corrected_word = pipeline.predict([text])[0]
        image_path = word_to_image.get(corrected_word)
        image_base64 = get_image_base64(image_path) if image_path else None

        response = {
            'spoken_word': text,
            'corrected_word': corrected_word,
            'image': image_base64
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'No speech recognized'}), 400

if __name__ == '__main__':
    app.run(debug=True)
