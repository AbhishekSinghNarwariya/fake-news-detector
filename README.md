# 📰 AI Fake News Detector

An AI-powered Fake News Detection System that classifies news articles as **Fake** or **Real** using **Natural Language Processing (NLP)** and **Machine Learning**. The application features a Flask REST API and a responsive web interface for real-time predictions.

---

## 🚀 Features

* Fake & Real News Classification
* Confidence Score for Predictions
* Text Preprocessing (Cleaning, Stopword Removal, Lemmatization)
* TF-IDF Feature Extraction
* Logistic Regression Model
* Flask REST API
* Responsive User Interface
* Dark Mode Support
* Prediction History
* Dashboard Statistics
* Copy Prediction Result
* Loading Indicator & Error Handling

---

## 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Flask
* Flask-CORS

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* NLTK
* Joblib

---

## 📁 Project Structure

```text
fake-news-detector/
│
├── backend/
│   ├── app.py
│   ├── predict.py
│   ├── train_model.py
│   ├── model/
│   └── dataset/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the Flask server:

```bash
python backend/app.py
```

Open `frontend/index.html` in your browser.

---

## 🌐 API Endpoints

| Method | Endpoint   | Description                          |
| ------ | ---------- | ------------------------------------ |
| GET    | `/`        | API status                           |
| GET    | `/health`  | Health check                         |
| POST   | `/predict` | Predict whether news is Fake or Real |

### Example Request

```json
{
  "news": "Scientists discovered a new species of butterfly in the Amazon rainforest."
}
```

---

## 📈 Future Enhancements

* Explainable AI (Prediction Explanation)
* News URL Analysis
* Model Comparison Dashboard
* Export Prediction History
* AI-Based Recommendations

---

## 👨‍💻 Author

**Abhishek Singh Narwariya**

B.Tech Student | IT(IOT)

---

## ⭐ Acknowledgement

If you found this project useful, consider giving it a **⭐ Star** on GitHub.
