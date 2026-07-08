import pandas as pd

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import joblib

# Load datasets
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# Display first 5 rows
print("Fake News Dataset:")
print(fake_df.head())

print("\n" + "=" * 60 + "\n")

print("True News Dataset:")
print(true_df.head())

print("\nFake Dataset Shape:")
print(fake_df.shape)

print("\nTrue Dataset Shape:")
print(true_df.shape)

print("\n" + "=" * 60)
print("Fake Dataset Information")
print("=" * 60)

fake_df.info()

print("\n" + "=" * 60)
print("True Dataset Information")
print("=" * 60)

true_df.info()

print("\n" + "=" * 60)
print("Missing Values in Fake Dataset")
print("=" * 60)

print(fake_df.isnull().sum())

print("\n" + "=" * 60)
print("Missing Values in True Dataset")
print("=" * 60)

print(true_df.isnull().sum())

print("\n" + "=" * 60)
print("Duplicate Rows in Fake Dataset")
print("=" * 60)

print(fake_df.duplicated().sum())

print("\n" + "=" * 60)
print("Duplicate Rows in True Dataset")
print("=" * 60)

print(true_df.duplicated().sum())

print("\n" + "=" * 60)
print("Statistical Summary of Fake Dataset")
print("=" * 60)

print(fake_df.describe(include='all'))

print("\n" + "=" * 60)
print("Statistical Summary of True Dataset")
print("=" * 60)

print(true_df.describe(include='all'))

print("\n" + "=" * 60)
print("Unique Subjects in Fake Dataset")
print("=" * 60)

print("Number of Unique Subjects:", fake_df["subject"].nunique())

print("\nSubject Distribution:")
print(fake_df["subject"].value_counts())

print("\n" + "=" * 60)
print("Unique Subjects in True Dataset")
print("=" * 60)

print("Number of Unique Subjects:", true_df["subject"].nunique())

print("\nSubject Distribution:")
print(true_df["subject"].value_counts())

print("\n" + "=" * 60)
print("Unique Subjects in Fake Dataset")
print("=" * 60)

print("Number of Unique Subjects:", fake_df["subject"].nunique())

print("\nSubject Distribution:")
print(fake_df["subject"].value_counts())

print("\n" + "=" * 60)
print("Unique Subjects in True Dataset")
print("=" * 60)

print("Number of Unique Subjects:", true_df["subject"].nunique())

print("\nSubject Distribution:")
print(true_df["subject"].value_counts())

print("\n" + "=" * 60)
print("Adding Labels to the Datasets")
print("=" * 60)

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

print("\nFake Dataset Labels:")
print(fake_df[["title", "label"]].head())

print("\nTrue Dataset Labels:")
print(true_df[["title", "label"]].head())

print("\n" + "=" * 60)
print("Merging Fake and True Datasets")
print("=" * 60)

# Merge datasets
news_df = pd.concat([fake_df, true_df], ignore_index=True)

print("\nMerged Dataset Shape:")
print(news_df.shape)

print("\nFirst 5 Rows of Merged Dataset:")
print(news_df.head())

print("\n" + "=" * 60)
print("Shuffling the Dataset")
print("=" * 60)

# Shuffle the dataset
news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset Shuffled Successfully!")

print("\nFirst 5 Rows After Shuffling:")
print(news_df.head())

print("\nLabel Distribution After Shuffling:")
print(news_df["label"].value_counts())

print("\n" + "=" * 60)
print("Removing Unnecessary Columns")
print("=" * 60)

# Keep only required columns
news_df = news_df[["text", "label"]]

print("\nRemaining Columns:")
print(news_df.columns)

print("\nDataset Shape After Removing Columns:")
print(news_df.shape)

print("\nFirst 5 Rows:")
print(news_df.head())

print("\n" + "=" * 60)
print("Creating Text Cleaning Function")
print("=" * 60)

def clean_text(text):
    """
    Clean a news article by:
    - Converting to lowercase
    - Removing URLs
    - Removing HTML tags
    - Removing punctuation
    - Removing numbers
    - Removing extra spaces
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

print("✅ Text cleaning function created successfully!")


print("\n" + "=" * 60)
print("Applying Text Cleaning to Dataset")
print("=" * 60)

# Apply cleaning function
news_df["clean_text"] = news_df["text"].apply(clean_text)

print("\n✅ Text cleaning completed successfully!")

print("\nOriginal Text:")
print(news_df["text"].iloc[0])

print("\n" + "=" * 60)

print("Cleaned Text:")
print(news_df["clean_text"].iloc[0])


print("\n" + "=" * 60)
print("Downloading NLTK Resources")
print("=" * 60)

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("punkt_tab")

print("\n✅ NLTK resources downloaded successfully!")

print("\n" + "=" * 60)
print("Creating Tokenization Function")
print("=" * 60)

def tokenize_text(text):
    """
    Convert cleaned text into a list of words.
    """
    return word_tokenize(text)

print("✅ Tokenization function created successfully!")

print("\n" + "=" * 60)
print("Applying Tokenization")
print("=" * 60)

news_df["tokens"] = news_df["clean_text"].apply(tokenize_text)

print("\n✅ Tokenization completed successfully!")

print("\nCleaned Text:")
print(news_df["clean_text"].iloc[0])

print("\nTokens:")
print(news_df["tokens"].iloc[0])


print("\n" + "=" * 60)
print("Preparing Stopwords")
print("=" * 60)

stop_words = set(stopwords.words("english"))

print(f"✅ Loaded {len(stop_words)} English stopwords.")


def remove_stopwords(tokens):
    """
    Remove common English stopwords from a list of tokens.
    """
    return [word for word in tokens if word not in stop_words]

print("✅ Stopword removal function created successfully!")


print("\n" + "=" * 60)
print("Removing Stopwords")
print("=" * 60)

news_df["filtered_tokens"] = news_df["tokens"].apply(remove_stopwords)

print("✅ Stopwords removed successfully!")

print("\nOriginal Tokens:")
print(news_df["tokens"].iloc[0])

print("\nFiltered Tokens:")
print(news_df["filtered_tokens"].iloc[0])

print("\n" + "=" * 60)
print("Preparing Lemmatizer")
print("=" * 60)

lemmatizer = WordNetLemmatizer()

print("✅ WordNet Lemmatizer initialized successfully!")


def lemmatize_tokens(tokens):
    """
    Convert each token to its base form.
    """
    return [lemmatizer.lemmatize(word) for word in tokens]

print("✅ Lemmatization function created successfully!")


print("\n" + "=" * 60)
print("Applying Lemmatization")
print("=" * 60)

news_df["lemmatized_tokens"] = news_df["filtered_tokens"].apply(lemmatize_tokens)

print("✅ Lemmatization completed successfully!")

print("\nFiltered Tokens:")
print(news_df["filtered_tokens"].iloc[0])

print("\nLemmatized Tokens:")
print(news_df["lemmatized_tokens"].iloc[0])


print("\n" + "=" * 60)
print("Creating Final Preprocessed Text")
print("=" * 60)

def join_tokens(tokens):
    """
    Join a list of tokens into a single string.
    """
    return " ".join(tokens)

print("✅ Join function created successfully!")


def preprocess_text(text):
    """
    Apply the complete preprocessing pipeline used during training.
    """
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    text = join_tokens(tokens)
    return text


news_df["processed_text"] = news_df["lemmatized_tokens"].apply(join_tokens)

print("✅ Final processed text created successfully!")

print("\nLemmatized Tokens:")
print(news_df["lemmatized_tokens"].iloc[0])

print("\nProcessed Text:")
print(news_df["processed_text"].iloc[0])


print("\n" + "=" * 60)
print("Preparing Features and Labels")
print("=" * 60)

X = news_df["processed_text"]
y = news_df["label"]

print("✅ Features and labels prepared successfully!")
print(f"Total samples: {len(X)}")


print("\n" + "=" * 60)
print("Splitting Dataset")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Dataset split successfully!")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\n" + "=" * 60)
print("Creating TF-IDF Vectorizer")
print("=" * 60)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

print("✅ TF-IDF Vectorizer created successfully!")


print("\n" + "=" * 60)
print("Applying TF-IDF to Training Data")
print("=" * 60)

X_train_tfidf = vectorizer.fit_transform(X_train)

print("✅ Training data vectorized successfully!")


print("\n" + "=" * 60)
print("Applying TF-IDF to Testing Data")
print("=" * 60)

X_test_tfidf = vectorizer.transform(X_test)

print("✅ Testing data vectorized successfully!")


print("\nTraining Matrix Shape:", X_train_tfidf.shape)
print("Testing Matrix Shape:", X_test_tfidf.shape)


print("\n" + "=" * 60)
print("Creating Logistic Regression Model")
print("=" * 60)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("✅ Logistic Regression model created successfully!")

print("\n" + "=" * 60)
print("Training Logistic Regression Model")
print("=" * 60)

model.fit(X_train_tfidf, y_train)

print("✅ Model trained successfully!")

print("\n" + "=" * 60)
print("Making Predictions")
print("=" * 60)

y_pred = model.predict(X_test_tfidf)

print("✅ Predictions generated successfully!")

print("\nFirst 10 Actual Labels:")
print(y_test.iloc[:10].tolist())

print("\nFirst 10 Predicted Labels:")
print(y_pred[:10].tolist())


print("\n" + "=" * 60)
print("Evaluating Model")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Accuracy Percentage: {accuracy * 100:.2f}%")


print("\n" + "=" * 60)
print("Generating Confusion Matrix")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

print("✅ Confusion Matrix Generated Successfully!\n")
print(cm)


tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")


print("\n" + "=" * 60)
print("Classification Report")
print("=" * 60)

report = classification_report(y_test, y_pred)

print("Class 0 = Fake News")
print("Class 1 = Real News\n")

print(report)



print("\n" + "=" * 60)
print("Saving Model")
print("=" * 60)

joblib.dump(model, "backend/model/fake_news_model.pkl")

print("✅ Model saved successfully!")


joblib.dump(vectorizer, "backend/model/tfidf_vectorizer.pkl")

print("✅ TF-IDF Vectorizer saved successfully!")


import os

print("\nSaved Files:")

print(
    "Model Exists:",
    os.path.exists("backend/model/fake_news_model.pkl")
)

print(
    "Vectorizer Exists:",
    os.path.exists("backend/model/tfidf_vectorizer.pkl")
)


print("\n" + "=" * 60)
print("Loading Saved Model")
print("=" * 60)

loaded_model = joblib.load("backend/model/fake_news_model.pkl")

print("✅ Model loaded successfully!")

loaded_vectorizer = joblib.load(
    "backend/model/tfidf_vectorizer.pkl"
)

print("✅ TF-IDF Vectorizer loaded successfully!")


print("\nLoaded Objects:")

print(type(loaded_model))
print(type(loaded_vectorizer))


print("\n" + "=" * 60)
print("Creating Prediction Function")
print("=" * 60)

def predict_news(news_text):
    """
    Predict whether a news article is Fake or Real.
    Returns prediction label and confidence.
    """

    if news_text is None:
        return {
            "error": "News text cannot be None."
        }

    if not isinstance(news_text, str):
        return {
            "error": "News text must be a string."
        }

    if news_text.strip() == "":
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
        "confidence": confidence,
        "processed_text": processed
    }

print("\n" + "=" * 60)
print("Testing Prediction Function")
print("=" * 60)

sample_news = """
The government announced a new education policy to improve
digital learning in schools across the country.
"""

result = predict_news(sample_news)

print("\nOriginal News:")
print(sample_news)

print("\nProcessed News:")
print(result["processed_text"])

print(f"\nPrediction : {result['prediction']}")
print(f"Confidence : {result['confidence']:.2f}%")

test_news = [
    "The government announced a new education policy to improve digital learning in schools.",
    "Scientists discovered a new species of butterfly in the Amazon rainforest.",
    "Aliens have secretly taken control of the world's governments according to anonymous sources.",
    "The cricket team won the championship after a thrilling final match.",
    "A miracle pill can make anyone lose 20 kilograms in just one week without exercise."
]


print("\n" + "=" * 60)
print("Testing Multiple News Articles")
print("=" * 60)

for i, news in enumerate(test_news, start=1):

    result = predict_news(news)

    print(f"\nArticle {i}")
    print("-" * 40)
    print(news)
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2f}%")


print("\n" + "=" * 60)
print("Testing with Real Dataset Articles")
print("=" * 60)

dataset_samples = [
    ("Real News", true_df["text"].iloc[0]),
    ("Fake News", fake_df["text"].iloc[0])
]

for actual_label, article in dataset_samples:

    result = predict_news(article)

    print("\n" + "-" * 60)
    print(f"Actual Label : {actual_label}")
    print(f"Predicted    : {result['prediction']}")
    print(f"Confidence   : {result['confidence']:.2f}%")

    print("\nArticle Preview:")
    print(article[:300] + "...")


print("\n" + "=" * 60)
print("Testing Invalid Inputs")
print("=" * 60)

invalid_inputs = [
    "",
    "     ",
    None,
    12345
]

for item in invalid_inputs:
    result = predict_news(item)

    print("\nInput:", repr(item))
    print("Output:", result)