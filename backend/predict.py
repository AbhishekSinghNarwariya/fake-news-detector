import joblib
import re
import string
from pathlib import Path

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================================
# Load Model & Vectorizer
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

loaded_model = joblib.load(MODEL_PATH)
loaded_vectorizer = joblib.load(VECTORIZER_PATH)


# ==========================================================
# Initialize NLP Tools
# ==========================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================================
# Text Preprocessing
# ==========================================================

def preprocess_text(text):

    text = clean_text(text)

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ==========================================================
# Prediction Function
# ==========================================================

def predict_news(news_text):

    if news_text is None:
        return {
            "error": "News text cannot be None."
        }

    if not isinstance(news_text, str):
        return {
            "error": "News text must be a string."
        }

    if not news_text.strip():
        return {
            "error": "News text cannot be empty."
        }

    processed = preprocess_text(news_text)

    vector = loaded_vectorizer.transform([processed])

    prediction = loaded_model.predict(vector)[0]

    probabilities = loaded_model.predict_proba(vector)[0]

    if prediction == 0:

        label = "Fake News"

        confidence = probabilities[0] * 100

    else:

        label = "Real News"

        confidence = probabilities[1] * 100

    return {

        "prediction": label,

        "confidence": round(float(confidence), 2),

        "processed_text": processed

    }


# ==========================================================
# Test Prediction
# ==========================================================

if __name__ == "__main__":

    sample_news = """
    Scientists discovered a new species of butterfly
    in the Amazon rainforest after years of research.
    """

    result = predict_news(sample_news)

    print("\nPrediction Result")
    print("=" * 40)

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Processed  : {result['processed_text']}")