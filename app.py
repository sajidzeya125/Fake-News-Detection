from flask import Flask, request, render_template
import joblib
import re
import string

app = Flask(__name__)

# Load your model & vectorizer
model = joblib.load('fake-news-model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_text(text):
    # Basic cleaning to replicate training
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    processed = preprocess_text(news_text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    label = 'Fake News' if prediction == 0 else 'Real News'
    return render_template('result.html', prediction=label)

if __name__ == "__main__":
    app.run(debug=True)