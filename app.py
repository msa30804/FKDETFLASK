from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load pre-trained model and vectorizer
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Empty input'}), 400

        cleaned_text = clean_text(text)
        text_vec = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vec)[0]
        confidence = model.predict_proba(text_vec)[0].max()

        result = {
            'prediction': 'Fake' if prediction == 1 else 'True',
            'confidence': f"{confidence * 100:.1f}%"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
