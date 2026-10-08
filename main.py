"""
Main Command Line Interface for Syntecxhub Sentiment Analysis Tool.

Author: Surag M S
Project: Syntecxhub AI Internship - Project 2
"""

import sys
from pathlib import Path

import config
from predict import load_model_and_vectorizer, predict_batch, predict_sentiment
from train import train_pipeline


def print_banner() -> None:
    """Print project banner and author header."""
    print("\n" + "=" * 50)
    print("       SYNTECXHUB AI INTERNSHIP PROJECT")
    print("           SENTIMENT ANALYSIS TOOL")
    print("=" * 50)
    print("Author: Surag M S")
    print("Domain: Artificial Intelligence (Project 2)")
    print("=" * 50)


def print_menu() -> None:
    """Display interactive CLI menu options."""
    print("\nMENU OPTIONS:")
    print("1. Train Model")
    print("2. Analyze Sentiment (Single Text)")
    print("3. View Model Metrics")
    print("4. Batch Analyze Text (from CSV)")
    print("5. Exit")


def handle_train_model() -> None:
    """Option 1: Execute model training pipeline."""
    print("\n[INFO] Starting Model Training Pipeline...\n")
    try:
        train_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Model training failed: {e}")


def handle_analyze_sentiment() -> None:
    """Option 2: Analyze sentiment for custom user text."""
    # Check model existence
    try:
        model, vectorizer = load_model_and_vectorizer()
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}")
        return
    except Exception as err:
        print(f"\n[ERROR] Could not load model: {err}")
        return

    print("\n--- SINGLE TEXT SENTIMENT ANALYSIS ---")
    user_input = input("Enter text to analyze:\n> ").strip()

    if not user_input:
        print("[WARNING] Empty input provided. Please enter valid text.")
        return

    try:
        result = predict_sentiment(user_input, model=model, vectorizer=vectorizer)

        print("\n----------------------------------------")
        print(f"Text       : {result['text']}")
        print(f"Cleaned    : {result['cleaned_text']}")
        print(f"Sentiment  : {result['sentiment']}")
        print(f"Confidence : {result['confidence']:.2f}%")
        print("----------------------------------------")
        print("Class Probabilities:")
        for cls_name, prob in result["probabilities"].items():
            highlight = " <-- Predicted" if cls_name == result["sentiment"] else ""
            print(f"  - {cls_name:<10}: {prob:>6.2f}%{highlight}")
        print("----------------------------------------")
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")


def handle_view_metrics() -> None:
    """Option 3: View evaluation metrics from saved report."""
    if not config.METRICS_PATH.exists():
        print(
            "\n[ERROR] Metrics file not found.\n"
            "Please train the model first using option 1."
        )
        return

    try:
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        print("\n" + content)
    except Exception as e:
        print(f"[ERROR] Could not read metrics file: {e}")


def handle_batch_prediction() -> None:
    """Option 4: Run batch prediction on CSV file."""
    try:
        model, vectorizer = load_model_and_vectorizer()
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}")
        return

    print("\n--- BATCH SENTIMENT ANALYSIS ---")
    input_path_str = input(
        f"Enter input CSV file path [Default: {config.INPUT_TEXTS_PATH}]:\n> "
    ).strip()
    input_path = Path(input_path_str) if input_path_str else config.INPUT_TEXTS_PATH

    output_path_str = input(
        f"Enter output CSV file path [Default: {config.PREDICTIONS_PATH}]:\n> "
    ).strip()
    output_path = Path(output_path_str) if output_path_str else config.PREDICTIONS_PATH

    try:
        results_df = predict_batch(
            input_file=input_path,
            output_file=output_path,
            model=model,
            vectorizer=vectorizer,
        )
        print(f"\n[SUCCESS] Batch analysis completed successfully!")
        print(f"Processed {len(results_df)} texts.")
        print(f"Results saved to: {output_path}")

        print("\nSample Preview:")
        print(results_df.head(5).to_string(index=False))
    except Exception as e:
        print(f"[ERROR] Batch analysis failed: {e}")


def main() -> None:
    """Main application loop."""
    print_banner()

    while True:
        print_menu()
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            handle_train_model()
        elif choice == "2":
            handle_analyze_sentiment()
        elif choice == "3":
            handle_view_metrics()
        elif choice == "4":
            handle_batch_prediction()
        elif choice == "5":
            print("\nThank you for using Syntecxhub Sentiment Analysis Tool. Goodbye!")
            sys.exit(0)
        else:
            print("\n[INVALID CHOICE] Please enter a valid number between 1 and 5.")


if __name__ == "__main__":
    main()
