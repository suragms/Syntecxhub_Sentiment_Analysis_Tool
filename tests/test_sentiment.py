"""
Unit Tests for Syntecxhub Sentiment Analysis Tool.

Author: Surag M S
Project: Syntecxhub AI Internship - Project 2
"""

from pathlib import Path
import unittest
import pandas as pd

import config
from predict import load_model_and_vectorizer, predict_batch, predict_sentiment
from preprocess import preprocess_text
from train import load_dataset, train_pipeline


class TestSentimentAnalysisTool(unittest.TestCase):
    """Test suite covering preprocessing, data loading, training, prediction, and edge cases."""

    def test_preprocess_text_basic(self):
        """Test standard preprocessing transformations."""
        raw_text = "I LOVE this product!!! Visit https://example.com or email test@example.com @user123 #Great"
        cleaned = preprocess_text(raw_text)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("test@example.com", cleaned)
        self.assertNotIn("@user123", cleaned)
        self.assertIn("great", cleaned)
        self.assertIn("love", cleaned)
        self.assertEqual(cleaned, cleaned.lower())

    def test_preprocess_text_edge_cases(self):
        """Test preprocessing edge cases (empty string, None, numbers, whitespace)."""
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text(None), "")
        self.assertEqual(preprocess_text("   "), "")
        self.assertEqual(preprocess_text("123 456"), "")

    def test_dataset_structure(self):
        """Test dataset CSV existence, non-emptiness, and required columns."""
        self.assertTrue(config.DATASET_PATH.exists(), "Dataset file should exist.")
        df = load_dataset(config.DATASET_PATH)
        self.assertFalse(df.empty, "Dataset should not be empty.")
        self.assertIn("text", df.columns, "Dataset must contain 'text' column.")
        self.assertIn("sentiment", df.columns, "Dataset must contain 'sentiment' column.")
        self.assertGreaterEqual(
            len(df["sentiment"].unique()), 2, "Dataset should have at least 2 classes."
        )

    def test_train_pipeline(self):
        """Test execution of training pipeline and artifact generation."""
        results = train_pipeline()
        self.assertIn("best_model_name", results)
        self.assertIn("metrics", results)
        self.assertTrue(config.MODEL_PATH.exists(), "Trained model pkl should exist.")
        self.assertTrue(config.VECTORIZER_PATH.exists(), "Vectorizer pkl should exist.")
        self.assertTrue(config.METRICS_PATH.exists(), "Metrics text file should exist.")
        self.assertTrue(
            config.CONFUSION_MATRIX_PATH.exists(), "Confusion matrix image should exist."
        )

    def test_model_loading(self):
        """Test loading saved model and vectorizer."""
        model, vectorizer = load_model_and_vectorizer()
        self.assertIsNotNone(model)
        self.assertIsNotNone(vectorizer)

    def test_prediction_positive_sentence(self):
        """Test prediction for a strongly positive sentence."""
        model, vectorizer = load_model_and_vectorizer()
        res = predict_sentiment(
            "I really love this product, it is amazing!", model=model, vectorizer=vectorizer
        )
        self.assertEqual(res["sentiment"], "Positive")
        self.assertGreater(res["confidence"], 0.0)

    def test_prediction_negative_sentence(self):
        """Test prediction for a strongly negative sentence."""
        model, vectorizer = load_model_and_vectorizer()
        res = predict_sentiment(
            "This was a terrible experience, complete waste of money!",
            model=model,
            vectorizer=vectorizer,
        )
        self.assertEqual(res["sentiment"], "Negative")
        self.assertGreater(res["confidence"], 0.0)

    def test_prediction_neutral_sentence(self):
        """Test prediction for a neutral sentence."""
        model, vectorizer = load_model_and_vectorizer()
        res = predict_sentiment(
            "The product arrived yesterday afternoon.", model=model, vectorizer=vectorizer
        )
        self.assertEqual(res["sentiment"], "Neutral")

    def test_missing_model_error(self):
        """Test error handling when loading non-existent model files."""
        fake_path = Path("models/non_existent_model.pkl")
        with self.assertRaises(FileNotFoundError):
            load_model_and_vectorizer(model_path=fake_path)

    def test_batch_prediction(self):
        """Test batch prediction function and output file creation."""
        model, vectorizer = load_model_and_vectorizer()
        results_df = predict_batch(model=model, vectorizer=vectorizer)
        self.assertIn("predicted_sentiment", results_df.columns)
        self.assertIn("confidence", results_df.columns)
        self.assertTrue(
            config.PREDICTIONS_PATH.exists(), "Predictions CSV should exist."
        )


if __name__ == "__main__":
    unittest.main()
