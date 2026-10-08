"""
Prediction Module for Sentiment Analysis Tool.

Author: Surag M S
Project: Syntecxhub AI Internship - Project 2
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import pandas as pd

import config
from preprocess import preprocess_text


def load_model_and_vectorizer(
    model_path: Optional[Union[str, Path]] = None,
    vectorizer_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Any]:
    """Load the trained machine learning model and TF-IDF vectorizer.

    Args:
        model_path: Path to saved model file (.pkl).
        vectorizer_path: Path to saved vectorizer file (.pkl).

    Returns:
        Tuple[Any, Any]: (loaded_model, loaded_vectorizer)

    Raises:
        FileNotFoundError: If model or vectorizer files are missing.
    """
    model_p = Path(model_path) if model_path else config.MODEL_PATH
    vectorizer_p = Path(vectorizer_path) if vectorizer_path else config.VECTORIZER_PATH

    if not model_p.exists() or not vectorizer_p.exists():
        raise FileNotFoundError(
            "Model files were not found.\n"
            "Please train the model first using option 1 or run 'python train.py'."
        )

    try:
        model = joblib.load(model_p)
        vectorizer = joblib.load(vectorizer_p)
        return model, vectorizer
    except Exception as e:
        raise RuntimeError(f"Failed to load trained model artifacts: {e}")


def predict_sentiment(
    text: str,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Predict sentiment and calculate class probabilities for a given text.

    Args:
        text (str): Input text string to classify.
        model (Optional[Any]): Pre-loaded scikit-learn model object.
        vectorizer (Optional[Any]): Pre-loaded TfidfVectorizer object.

    Returns:
        Dict[str, Any]: Dictionary containing prediction details:
            - text: Original raw text
            - cleaned_text: Cleaned text
            - sentiment: Predicted class label (capitalized)
            - confidence: Percentage probability float
            - probabilities: Dictionary mapping label to percentage
    """
    if text is None or not isinstance(text, str) or text.strip() == "":
        return {
            "text": text if text is not None else "",
            "cleaned_text": "",
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0},
        }

    # Load artifacts if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer()

    # Preprocess input text
    cleaned_text = preprocess_text(text)
    if cleaned_text.strip() == "":
        cleaned_text = text.lower().strip()

    # Vectorize cleaned text
    text_tfidf = vectorizer.transform([cleaned_text])

    # Predict class and probabilities
    predicted_label = model.predict(text_tfidf)[0]

    probabilities_dict: Dict[str, float] = {}
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(text_tfidf)[0]
        classes = model.classes_
        for cls, prob in zip(classes, probs):
            formatted_cls = str(cls).strip().capitalize()
            probabilities_dict[formatted_cls] = round(float(prob) * 100, 2)

        # Get confidence score for predicted label
        pred_index = list(classes).index(predicted_label)
        confidence = round(float(probs[pred_index]) * 100, 2)
    else:
        # Fallback if model doesn't support predict_proba
        formatted_label = str(predicted_label).capitalize()
        probabilities_dict[formatted_label] = 100.0
        confidence = 100.0

    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "sentiment": str(predicted_label).capitalize(),
        "confidence": confidence,
        "probabilities": probabilities_dict,
    }


def predict_batch(
    input_file: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    model: Optional[Any] = None,
    vectorizer: Optional[Any] = None,
) -> pd.DataFrame:
    """Run batch sentiment analysis on an input CSV file and save results.

    Args:
        input_file: Path to input CSV containing a 'text' column.
        output_file: Path to destination CSV file for predictions.
        model: Optional pre-loaded model.
        vectorizer: Optional pre-loaded vectorizer.

    Returns:
        pd.DataFrame: DataFrame containing batch predictions.

    Raises:
        FileNotFoundError: If input file is missing.
        ValueError: If input file lacks 'text' column.
    """
    in_path = Path(input_file) if input_file else config.INPUT_TEXTS_PATH
    out_path = Path(output_file) if output_file else config.PREDICTIONS_PATH

    if not in_path.exists():
        raise FileNotFoundError(f"Input batch CSV file not found at: {in_path}")

    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        raise ValueError(f"Error reading input CSV file: {e}")

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer()

    sentiments = []
    confidences = []

    for text in df["text"]:
        res = predict_sentiment(str(text), model=model, vectorizer=vectorizer)
        sentiments.append(res["sentiment"])
        confidences.append(round(res["confidence"] / 100.0, 4))  # Save confidence as ratio (0.0 to 1.0)

    results_df = df.copy()
    results_df["predicted_sentiment"] = sentiments
    results_df["confidence"] = confidences

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    return results_df


if __name__ == "__main__":
    # Quick prediction test
    test_sentence = "I really enjoyed using this application!"
    try:
        result = predict_sentiment(test_sentence)
        print("Test Text  :", result["text"])
        print("Sentiment  :", result["sentiment"])
        print(f"Confidence : {result['confidence']:.2f}%")
        print("Probabilities:", result["probabilities"])
    except Exception as err:
        print(f"Prediction failed: {err}")
