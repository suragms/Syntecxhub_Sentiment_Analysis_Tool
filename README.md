# Sentiment Analysis Tool

**Syntecxhub Artificial Intelligence Internship - Project 2**

**Author:** Surag M S

---

## Overview

The **Sentiment Analysis Tool** is a complete, machine learning-based Natural Language Processing (NLP) application built in Python. It classifies text inputs—such as product reviews, user comments, or social media tweets—into sentiment categories (**Positive**, **Negative**, or **Neutral**).

This project uses **TF-IDF Feature Extraction** combined with **Logistic Regression** and **Multinomial Naive Bayes** classifiers to deliver accurate predictions along with class probability confidence scores. It features an interactive command-line interface (CLI), automated model evaluation, visualization, batch processing, and persistent model serialization.

---

## Features

- **Text Preprocessing:** Automated text cleaning including lowercasing, removal of URLs, email addresses, `@mentions`, punctuation, numbers, and excess whitespace.
- **TF-IDF Vectorization:** Converts raw textual data into numerical feature vectors.
- **Logistic Regression Classifier:** Primary machine learning classifier for accurate sentiment prediction.
- **Naive Bayes Comparison:** Automatically trains and compares Logistic Regression with Multinomial Naive Bayes.
- **Multi-Class Sentiment Classification:** Supports Positive, Negative, and Neutral sentiment detection.
- **Comprehensive Model Evaluation:** Calculates Accuracy, Precision, Recall, F1 Score, Classification Report, and Confusion Matrix.
- **Confusion Matrix Visualization:** Generates and saves a graphical confusion matrix plot (`results/confusion_matrix.png`).
- **Interactive CLI:** User-friendly menu interface to train models, test single sentences, view metrics, or run batch analysis.
- **Confidence Score:** Calculates prediction probability percentage for each sentiment class.
- **Batch Processing:** Processes multiple texts from a CSV file (`data/input_texts.csv`) and exports predictions to `results/predictions.csv`.
- **Model Serialization:** Saves trained models and vectorizers using `joblib` for immediate offline prediction without retraining.

---

## Technologies Used

- **Python 3.10+**
- **Pandas:** Data loading, manipulation, and CSV export.
- **NumPy:** Numerical computations.
- **Scikit-learn:** TF-IDF feature extraction, model training, evaluation metrics, train-test splitting.
- **Joblib:** Model and vectorizer serialization.
- **Matplotlib:** Confusion matrix plot rendering.
- **Re & Pathlib:** Regex processing and standard file system operations.
- **Pytest / Unittest:** Automated software testing.

---

## Machine Learning Workflow

```
┌─────────────────────────┐
│   Raw Sentiment Data    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Text Preprocessing    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  TF-IDF Feature Extraction│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Train/Test Split     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Model Training & Compare│
│ (Logistic Reg vs NB)    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Model Evaluation &    │
│  Confusion Matrix Plot  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Sentiment Prediction &  │
│    Confidence Score     │
└─────────────────────────┘
```

---

## Project Structure

```
Syntecxhub_Sentiment_Analysis_Tool/
│
├── main.py                     # Main interactive CLI application
├── train.py                    # Training pipeline & model comparison module
├── predict.py                  # Single & batch sentiment prediction module
├── preprocess.py               # Reusable text preprocessing functions
├── evaluate.py                 # Evaluation metrics & confusion matrix plot helper
├── config.py                   # Central configuration & hyperparameters
│
├── data/
│   ├── sentiment_data.csv      # Sample labeled sentiment dataset
│   └── input_texts.csv         # Sample batch input text dataset
│
├── models/
│   ├── sentiment_model.pkl     # Trained sentiment classification model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer artifact
│
├── results/
│   ├── metrics.txt             # Saved model evaluation metrics & comparison report
│   ├── predictions.csv         # Exported batch prediction output
│   └── confusion_matrix.png    # Saved confusion matrix visualization chart
│
├── tests/
│   └── test_sentiment.py       # Unit test suite (unittest / pytest)
│
├── requirements.txt            # Project dependencies list
├── .gitignore                  # Git ignore file configuration
├── LICENSE                     # MIT Open Source License
└── README.md                   # Comprehensive project documentation
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/suragms/Syntecxhub_Sentiment_Analysis_Tool.git
cd Syntecxhub_Sentiment_Analysis_Tool
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- **Windows:**
  ```cmd
  venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Interactive CLI Mode
To start the application interface:
```bash
python main.py
```

### Train the Model Directly
To train the classifier, evaluate metrics, and save model artifacts directly:
```bash
python train.py
```

### Run Unit Tests
To verify all modules and tests:
```bash
python -m unittest discover tests
```
or
```bash
python -m pytest
```

---

## Example Prediction

### CLI Input:
```
Enter text to analyze:
> I really enjoyed using this application!
```

### Output:
```
----------------------------------------
Text       : I really enjoyed using this application!
Cleaned    : i really enjoyed using this application
Sentiment  : Positive
Confidence : 89.54%
----------------------------------------
Class Probabilities:
  - Positive  :  89.54% <-- Predicted
  - Negative  :   5.23%
  - Neutral   :   5.23%
----------------------------------------
```

---

## Evaluation Metrics

The model is evaluated using five primary performance metrics:

1. **Accuracy:** Overall proportion of correctly predicted sentiments out of total samples.
2. **Precision:** Ratio of correctly predicted positive observations to total predicted positive observations.
3. **Recall:** Ratio of correctly predicted positive observations to all observations in actual class.
4. **F1 Score:** Weighted harmonic mean of Precision and Recall.
5. **Confusion Matrix:** Matrix representation comparing ground truth targets against predicted classes.

---

## Dataset Format

The project relies on `data/sentiment_data.csv` formatted as follows:

| text | sentiment |
|---|---|
| "I really love this product, it works amazingly well!" | positive |
| "This was a terrible experience, totally disappointed." | negative |
| "The product is okay, nothing special but works." | neutral |

You can replace `data/sentiment_data.csv` with a larger dataset containing custom tweets or reviews.

---

## Viva Questions & Answers

1. **What is sentiment analysis?**  
   Sentiment analysis is an NLP technique used to analyze text and determine the emotional tone or attitude (positive, negative, neutral) expressed within it.

2. **What is NLP?**  
   Natural Language Processing (NLP) is a branch of Artificial Intelligence focused on enabling computers to understand, interpret, and process human language.

3. **What is text preprocessing?**  
   Text preprocessing cleans raw text by lowercasing, removing noise (URLs, punctuation, numbers, special characters), and normalizing white space before feature extraction.

4. **What is tokenization?**  
   Tokenization splits text into smaller units called tokens, such as individual words or sub-words, for computational processing.

5. **What is TF-IDF?**  
   Term Frequency-Inverse Document Frequency (TF-IDF) converts text into numerical feature vectors by weighting word frequency relative to how rare it is across documents.

6. **Why is TF-IDF used?**  
   TF-IDF highlights important, informative words while penalizing common words (e.g., "the", "is") that appear frequently across all documents.

7. **What is Logistic Regression?**  
   Logistic Regression is a supervised classification algorithm that calculates the probability of a target label using a logistic sigmoid function.

8. **Why did you choose Logistic Regression?**  
   Logistic Regression is fast, highly effective for sparse high-dimensional text data, interpretable, and provides calibrated probability scores via `predict_proba()`.

9. **What is Naive Bayes?**  
   Multinomial Naive Bayes is a probabilistic classifier based on Bayes' Theorem that assumes conditional independence between input features.

10. **What is training data?**  
    Training data is the subset of labeled data used to fit the model parameters during the training phase.

11. **What is testing data?**  
    Testing data is an unseen evaluation subset used to measure how well the trained model generalizes to new data.

12. **Why use train_test_split?**  
    It splits data into independent training and testing sets to prevent data leakage and assess model overfitting.

13. **What is accuracy?**  
    Accuracy measures the percentage of correct predictions out of all total test samples.

14. **What is precision?**  
    Precision measures how many of the positively predicted instances were actually correct.

15. **What is recall?**  
    Recall measures how many of the actual positive instances were correctly identified by the model.

16. **What is F1 score?**  
    F1 Score is the harmonic mean of precision and recall, serving as a balanced metric especially when classes are imbalanced.

17. **What is a confusion matrix?**  
    A confusion matrix is a table layout illustrating true positive, false positive, true negative, and false negative predictions across classes.

18. **What is overfitting?**  
    Overfitting occurs when a model learns noise and specific details of the training data too well, resulting in poor generalization on unseen test data.

19. **What is the purpose of random_state?**  
    Setting `random_state` ensures reproducibility by fixing the pseudo-random number generator seed across splits and training runs.

20. **How does the project predict sentiment?**  
    Raw text is cleaned by `preprocess.py`, vectorized by `TfidfVectorizer`, classified by `LogisticRegression`, and outputs the predicted sentiment with a confidence percentage.

---

## Future Improvements

- Incorporate larger real-world datasets (such as Twitter API data or Amazon customer reviews).
- Integrate advanced Transformer architectures (e.g., BERT, RoBERTa) for contextual embeddings.
- Deploy a web interface using **Flask**, **FastAPI**, or **Streamlit**.
- Expand support for multi-language sentiment detection.

---

## Internship Information

- **Organization:** Syntecxhub
- **Domain:** Artificial Intelligence
- **Project:** Sentiment Analysis Tool
- **Project Number:** 2
- **Author:** Surag M S

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
