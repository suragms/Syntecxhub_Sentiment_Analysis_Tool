"""
Configuration settings for Syntecxhub Sentiment Analysis Tool.

Author: Surag M S
Project: Syntecxhub AI Internship - Project 2
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Directory paths
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data file paths
DATASET_PATH = DATA_DIR / "sentiment_data.csv"
INPUT_TEXTS_PATH = DATA_DIR / "input_texts.csv"

# Saved model paths
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# Results file paths
METRICS_PATH = RESULTS_DIR / "metrics.txt"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

# Train/Test Split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF Vectorizer parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = "english"
TFIDF_NGRAM_RANGE = (1, 2)

# Model parameters
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

NAIVE_BAYES_PARAMS = {
    "alpha": 1.0,
}

# Supported sentiment classes
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]
