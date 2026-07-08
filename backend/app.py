from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_news

from dotenv import load_dotenv
import os
import logging

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# ==========================================================
# Flask App
# ==========================================================

app = Flask(__name__)

CORS(app)

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# Routes
# ==========================================================

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

        data = request.get_json()

        if not data:

            logger.warning("No JSON data received.")

            return jsonify({
                "success": False,
                "message": "No JSON data received."
            }), 400

        news = data.get("news")

        logger.info("Prediction request received.")

        if news is None or not str(news).strip():

            logger.warning("Empty news text received.")

            return jsonify({
                "success": False,
                "message": "News text cannot be empty."
            }), 400

        result = predict_news(news)

        if "error" in result:

            logger.error(result["error"])

            return jsonify({
                "success": False,
                "message": result["error"]
            }), 400

        logger.info("Prediction completed successfully.")

        return jsonify({
            "success": True,
            "message": "Prediction completed successfully.",
            "data": result
        }), 200

    except Exception as e:

        logger.exception("Unexpected server error")

        return jsonify({
            "success": False,
            "message": "Internal Server Error",
            "error": str(e)
        }), 500


# ==========================================================
# Run Application
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=DEBUG
    )