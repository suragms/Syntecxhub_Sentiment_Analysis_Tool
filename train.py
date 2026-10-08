"""
Training Pipeline for Sentiment Analysis Tool.

Author: Surag M S
Project: Syntecxhub AI Internship - Project 2
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import config
from evaluate import (
    calculate_metrics,
    generate_classification_report_str,
    generate_confusion_matrix,
    save_metrics,
)
from preprocess import preprocess_text


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and validate the sentiment dataset.

    Args:
        file_path (Path): Path to CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If dataset CSV file does not exist.
        ValueError: If dataset is empty or lacks required columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {file_path}. Please check the file path."
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV dataset: {e}")

    if df.empty:
        raise ValueError("Dataset is empty. Please provide a non-empty dataset.")

    required_columns = {"text", "sentiment"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain 'text' and 'sentiment' columns. Found: {list(df.columns)}"
        )

    # Drop rows with missing values in text or sentiment
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    unique_classes = df["sentiment"].unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset must contain at least 2 sentiment classes for training. Found: {unique_classes}"
        )

    return df


def train_pipeline() -> dict:
    """Run complete training, evaluation, comparison, and saving pipeline.

    Returns:
        dict: Summary of training results.
    """
    print("=" * 60)
    print("      SYNTECXHUB SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("=" * 60)
    print("Author: Surag M S\n")

    # Step 1: Load dataset
    print(f"[1/6] Loading dataset from: {config.DATASET_PATH}")
    df = load_dataset(config.DATASET_PATH)
    print(f"      Total dataset size: {len(df)} samples")
    print(f"      Class distribution:\n{df['sentiment'].value_counts().to_string(header=False)}\n")

    # Step 2: Clean and preprocess text
    print("[2/6] Cleaning and preprocessing text data...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # Filter out empty cleaned strings
    df = df[df["cleaned_text"].str.strip() != ""].copy()

    X = df["cleaned_text"]
    y = df["sentiment"]

    # Step 3: Train / Test Split
    print(f"[3/6] Splitting data (test_size={config.TEST_SIZE}, random_state={config.RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"      Training set size : {len(X_train)} samples")
    print(f"      Testing set size  : {len(X_test)} samples\n")

    # Step 4: Feature Extraction via TF-IDF
    print("[4/6] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        stop_words=config.TFIDF_STOP_WORDS,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"      TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} features\n")

    # Step 5: Model Training & Comparison
    print("[5/6] Training Machine Learning Classifiers...")

    # Model 1: Logistic Regression
    lr_model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    lr_metrics = calculate_metrics(y_test, y_pred_lr)

    # Model 2: Multinomial Naive Bayes
    nb_model = MultinomialNB(**config.NAIVE_BAYES_PARAMS)
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    nb_metrics = calculate_metrics(y_test, y_pred_nb)

    # Model Comparison Display
    comparison_table = (
        f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12}\n"
        f"{'-'*49}\n"
        f"{'Logistic Regression':<25} {lr_metrics['accuracy']:<12.4f} {lr_metrics['f1_score']:<12.4f}\n"
        f"{'Multinomial Naive Bayes':<25} {nb_metrics['accuracy']:<12.4f} {nb_metrics['f1_score']:<12.4f}"
    )

    print("\n========================================")
    print("MODEL COMPARISON")
    print("========================================")
    print(comparison_table)
    print("========================================\n")

    # Model Selection (Prefer Logistic Regression if tied)
    if nb_metrics["f1_score"] > lr_metrics["f1_score"]:
        best_model_name = "Multinomial Naive Bayes"
        best_model = nb_model
        best_metrics = nb_metrics
        y_pred_best = y_pred_nb
    else:
        best_model_name = "Logistic Regression"
        best_model = lr_model
        best_metrics = lr_metrics
        y_pred_best = y_pred_lr

    print(f"Selected Best Model: {best_model_name} (F1 Score: {best_metrics['f1_score']:.4f})\n")

    # Step 6: Detailed Evaluation & Saving Artifacts
    print("[6/6] Evaluating selected model and saving artifacts...")
    labels = sorted(list(set(y_test)))
    report_str = generate_classification_report_str(y_test, y_pred_best, target_names=labels)
    cm = generate_confusion_matrix(
        y_test, y_pred_best, labels=labels, output_path=config.CONFUSION_MATRIX_PATH
    )

    # Save metrics text report
    save_metrics(
        best_metrics,
        report_str,
        cm,
        labels,
        comparison_table,
        config.METRICS_PATH,
    )

    # Save model and vectorizer binaries
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)

    print(f"      Saved Model       : {config.MODEL_PATH}")
    print(f"      Saved Vectorizer  : {config.VECTORIZER_PATH}")
    print(f"      Saved Metrics     : {config.METRICS_PATH}")
    print(f"      Saved Confusion Matrix Plot: {config.CONFUSION_MATRIX_PATH}")

    print("\n==================================================")
    print("       TRAINING COMPLETED SUCCESSFULLY!")
    print("==================================================\n")

    return {
        "best_model_name": best_model_name,
        "metrics": best_metrics,
        "classification_report": report_str,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    try:
        train_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Model training failed: {e}", file=sys.stderr)
        sys.exit(1)
