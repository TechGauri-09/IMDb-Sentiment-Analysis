# 🎬 IMDb Movie Review Sentiment Analysis

I built this project during my Data Science internship at **Nettech India** — it reads a movie review and figures out whether the person liked the movie or not, just by analyzing the text.

---

## 📌 What this project does

Every day, thousands of people leave reviews on IMDb, and manually reading each one to gauge public opinion just isn't practical. This project automates that. I trained a model on 50,000 real IMDb reviews so it can look at a brand new review it's never seen before and tell you, with a confidence score, whether it's positive or negative.

- **Dataset:** 50,000 IMDb reviews, evenly split — 25,000 positive, 25,000 negative
- **Approach:** Clean the text → convert it into numbers using TF-IDF → train and compare models
- **Final Model:** Logistic Regression, landing at **89.13% accuracy**
- **Deployment:** Wrapped it in a Flask app so anyone can type a review and get a live prediction

---

## 📊 How well does it actually work?

| Model | Accuracy | F1-score |
|---|---|---|
| **Logistic Regression** ✅ | **89.13%** | **0.89** |
| Naive Bayes | 85.48% | 0.85 |

I trained both models on the same data just to see which one handles this better. Logistic Regression won out — not just on accuracy, but it was also more balanced across positive and negative reviews, so that's the one I went with.

---

## 🛠️ Built with

- **Language:** Python
- **Data Handling:** pandas, numpy
- **NLP:** NLTK for stopword removal and lemmatization
- **ML:** scikit-learn — TF-IDF, Logistic Regression, Naive Bayes
- **Deployment:** Flask, plain HTML/CSS/JS
- **Visualization:** matplotlib, seaborn

---

## ⚙️ The pipeline, step by step

1. **Cleaning the text** — real reviews are messy. I stripped out HTML leftovers, punctuation, numbers, converted everything to lowercase, removed common filler words (stopwords), and reduced words to their base form (lemmatization) so "loved" and "loving" aren't treated as different words.
2. **Turning text into numbers** — models can't read words, so I used TF-IDF to convert cleaned reviews into numerical vectors, keeping the 5,000 most meaningful terms.
3. **Training and comparing models** — trained Logistic Regression and Naive Bayes side by side on identical data to see which generalizes better.
4. **Evaluating properly** — didn't just look at accuracy; checked precision, recall, F1-score, and a confusion matrix to make sure the model wasn't just guessing one class more often.
5. **Making it usable** — a trained model sitting in a notebook doesn't help anyone, so I deployed it with Flask so it can actually be used through a browser.

---

## 📂 Project Structure

```
imdb-sentiment-analysis/
├── code.ipynb              # Full pipeline: cleaning, TF-IDF, training, evaluation
├── app.py                  # Flask backend serving the trained model
├── templates/
│   └── index.html          # Web interface for live predictions
├── IMDb_Sentiment_Analysis_Project_Report.pdf   # Full project report
└── README.md
```

---

## 🚀 Want to run it yourself?

```bash
git clone https://github.com/TechGauri-09/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

pip install flask scikit-learn nltk pandas

python app.py
```

Then just open `http://127.0.0.1:5000` in your browser and try it out.

One thing to keep in mind — run `code.ipynb` once first. That's what generates `sentiment_model.pkl` and `tfidf_vectorizer.pkl`, and `app.py` needs those files to actually load the trained model.

---

## 📈 A couple of test runs

| Review | Prediction | Confidence |
|---|---|---|
| "This movie was absolutely wonderful, I loved every minute!" | Positive | ~96% |
| "Waste of time, boring and poorly acted." | Negative | ~94% |

---

## 🔮 What I'd add next

- Swap TF-IDF for word embeddings (Word2Vec/GloVe) or try fine-tuning a BERT model via HuggingFace — should push accuracy higher
- Better handling for sarcasm and reviews with mixed sentiment, which the model currently struggles with
- Host it properly on Render or HuggingFace Spaces so it's not just running locally

---

## 👤 About me

**Gauri Thakare**
B.Sc. Data Science, Viva College — University of Mumbai
[LinkedIn](https://linkedin.com/in/gauri-thakare-aba165320) • [GitHub](https://github.com/TechGauri-09)

---

*Built this as part of my NLP internship at Nettech India — feel free to explore the code or reach out if you have questions!*
