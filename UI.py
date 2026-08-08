from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

UI = Flask(__name__)

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.replace(' br ', ' ')
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@UI.route('/')
def home():
    return render_template('index.html')

@UI.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')

    if not review.strip():
        return jsonify({'error': 'Empty review'}), 400

    cleaned = clean_text(review)
    vec = tfidf.transform([cleaned])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]

    confidence = float(max(probability)) * 100
    sentiment = 'Positive' if prediction == 1 else 'Negative'

    return jsonify({
        'sentiment': sentiment,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    UI.run(debug=True)