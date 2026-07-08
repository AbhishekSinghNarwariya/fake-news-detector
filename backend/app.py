from flask import Flask, request, jsonify
from predict import predict_news
import logging
from flask_cors import CORS

from dotenv import load_dotenv
import os

load_dotenv()

PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("FLASK_DEBUG", "False") == "True"

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@app.route("/")
def home():
    return jsonify({
        "success": True,
        "message": "Fake News Detection API is running successfully!"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "model": "loaded",
        "version": "1.0.0"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Check if JSON data exists
        if not data:
            logging.warning("No JSON data received.")
            return jsonify({
                "success": False,
                "message": "No JSON data received."
            }), 400

        # Get news text
        news = data.get("news")

        logging.info("Prediction request received.")

        # Validate news field
        if news is None or not str(news).strip():
            logging.warning("Empty news text received.")
            return jsonify({
                "success": False,
                "message": "News text cannot be empty."
            }), 400

        # Make prediction
        result = predict_news(news)

        # Handle prediction errors
        if "error" in result:
            logging.error(result["error"])
            return jsonify({
                "success": False,
                "message": result["error"]
            }), 400

        # Log successful prediction
        logging.info("Prediction completed successfully.")

        # Success response
        return jsonify({
            "success": True,
            "message": "Prediction completed successfully.",
            "data": result
        }), 200

    except Exception as e:
        logging.exception("An unexpected error occurred.")

        return jsonify({
            "success": False,
            "message": "Internal Server Error",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=DEBUG
    )